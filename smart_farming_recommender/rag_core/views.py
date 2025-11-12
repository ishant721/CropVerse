from .models import Report, ChatSession, ChatMessage
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, serializers
from .rag_pipeline import get_rag_pipeline, perform_tavily_search # Import perform_tavily_search
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from rest_framework.permissions import IsAuthenticated
import tempfile
import pytesseract
import base64
from PIL import Image
import io
import re # Import regex module

def process_image_to_text(image_file):
    """
    Uses pytesseract to perform OCR on an image file and extract text.
    """
    try:
        # Ensure the file pointer is at the beginning
        image_file.seek(0)
        img = Image.open(image_file)
        # Use pytesseract to do OCR on the image
        text = pytesseract.image_to_string(img)
        if text.strip():
            return f"The user provided an image containing the following text: {text}"
        else:
            return "User provided an image, but no text could be extracted."
    except Exception as e:
        print(f"Error processing image with pytesseract: {e}")
        return "User provided an image, but there was an error processing it."

@login_required
def ai_dashboard(request):
    return render(request, 'rag_core/ai_dashboard.html')

@login_required
def planning_page(request):
    # Fetch all reports for the current user, ordered by creation date
    user_reports = Report.objects.filter(user=request.user)
    context = {
        'reports': user_reports
    }
    return render(request, 'rag_core/planning.html', context)

class RAGChatView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user_question = request.data.get('question', '')
        image_files = request.FILES.getlist('images')
        session_id = request.data.get('session_id')

        if not user_question and not image_files:
            return Response({'error': 'Question or image is required.'}, status=status.HTTP_400_BAD_REQUEST)

        # --- Combine text and image data into a single message list ---
        message_parts = []
        if user_question:
            message_parts.append({"type": "text", "text": user_question})

        image_path = None
        if image_files:
            image_file = image_files[0] # For now, only handle one image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
                for chunk in image_file.chunks():
                    temp_image.write(chunk)
                image_path = temp_image.name
            
            # Add image to message parts for display
            image_file.seek(0) # Reset file pointer to the beginning
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            message_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            })

        if not message_parts:
            return Response({'error': 'No valid question or image content to process.'}, status=status.HTTP_400_BAD_REQUEST)

        # --- Session Management ---
        session = None
        title = None
        if session_id:
            session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        else:
            # Create a new session
            title = (user_question[:150] + '...') if len(user_question) > 150 else user_question
            session = ChatSession.objects.create(user=request.user, title=title)
        
        # Save user message
        # We need to decide how to save the multimodal message
        # For now, we'll just save the text part
        if user_question:
            ChatMessage.objects.create(session=session, message=user_question, is_user_message=True)

        # --- RAG Pipeline Execution ---
        try:
            rag_pipeline = get_rag_pipeline()
            inputs = {"question": message_parts, "image_path": image_path}
            final_state = rag_pipeline.invoke(inputs)
            
            if final_state is None:
                return Response({'error': 'The RAG pipeline did not produce a final result.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            answer = final_state.get("generation", "I can only help you with farmer related queries.")
            
            # Save AI message
            ChatMessage.objects.create(session=session, message=answer, is_user_message=False)

            response_data = {
                'answer': answer,
                'session_id': session.id,
            }
            if title: # Include title and session data if it's a new session
                response_data['new_session'] = {
                    'id': session.id,
                    'title': session.title
                }

            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            print(f"Error in RAG pipeline: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@login_required
def chat_page(request):
    # Fetch all chat sessions for the current user
    user_sessions = ChatSession.objects.filter(user=request.user)
    context = {
        'sessions': user_sessions
    }
    return render(request, 'rag_core/chat.html', context)

class GenerateReportView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        print("\n--- [GenerateReportView] Received POST request ---")
        user_report_query = request.data.get('user_report_query', '').strip()
        soil_type = request.data.get('soil_type', '').strip()
        budget = request.data.get('budget', '').strip()
        crop_preference = request.data.get('crop_preference', '').strip()
        land_area = request.data.get('land_area', '').strip()
        climate_zone = request.data.get('climate_zone', '').strip()
        additional_notes = request.data.get('additional_notes', '').strip()
        image_file = request.FILES.get('image') # Get uploaded image file

        image_description = ""
        if image_file:
            print("[GenerateReportView] Processing uploaded image.")
            image_description = process_image_to_text(image_file)
            if "error" in image_description.lower(): # Check if image processing failed
                print(f"[GenerateReportView] Image processing failed: {image_description}")
                return Response({'error': f"Image processing failed: {image_description}"}, status=status.HTTP_400_BAD_REQUEST)

        # ... (rest of the data collection and query construction) ...
        
        # Construct a comprehensive query for the RAG pipeline
        report_query_parts = []

        if user_report_query:
            report_query_parts.append(f"User's specific request: {user_report_query}")
        else:
            report_query_parts.append("Generate a detailed farming report and recommendations.")
            
        report_query_parts.append("Consider the following information (if provided):")

        if soil_type:
            report_query_parts.append(f"- Soil Type: {soil_type}")
        if budget:
            report_query_parts.append(f"- Budget: ${budget}")
        if crop_preference:
            report_query_parts.append(f"- Crop Preference: {crop_preference}")
        if land_area:
            report_query_parts.append(f"- Land Area: {land_area} acres")
        if climate_zone:
            report_query_parts.append(f"- Climate Zone: {climate_zone}")
        if additional_notes:
            report_query_parts.append(f"- Additional Notes: {additional_notes}")
        if image_description:
            report_query_parts.append(f"- Image Context: {image_description}")
        
        # ... (Tavily search logic is omitted for brevity but is still there) ...

        # Add instructions for comprehensive report generation
        report_query_parts.append(
            "The report should cover the entire process from cultivation to harvesting, "
            "including optimal crop selection, detailed cultivation practices, pest and disease management, "
            "irrigation strategies, fertilization plans, estimated costs, expected yields, "
            "and current market rates for the produce. Provide a comprehensive and actionable plan."
        )
        
        report_query = "\n".join(report_query_parts)
        print(f"[GenerateReportView] Constructed report query (first 100 chars): {report_query[:100]}")
        
        if not report_query.strip() or (len(report_query_parts) == 2 and "Generate a detailed farming report and recommendations." in report_query_parts[0] and "Consider the following information (if provided):" in report_query_parts[1]):
             print("[GenerateReportView] No information provided to generate a report.")
             return Response({'error': 'Please provide some information or a query to generate a report.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            print("[GenerateReportView] Getting RAG pipeline.")
            rag_pipeline = get_rag_pipeline()
            inputs = {"question": report_query}
            
            print("[GenerateReportView] Invoking RAG pipeline...")
            final_state = rag_pipeline.invoke(inputs)
            print("[GenerateReportView] RAG pipeline invocation finished.")
            
            if not final_state or "generation" not in final_state:
                print("[GenerateReportView] Error: Pipeline finished but no 'generation' in final state.")
                return Response({'error': 'Could not generate a detailed report.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            report_content = final_state["generation"]
            print("[GenerateReportView] Successfully generated report content.")

            # Generate a title for the report
            title = user_report_query if user_report_query else "Farming Report"
            title = (title[:197] + '...') if len(title) > 200 else title

            print(f"[GenerateReportView] Saving report with title: {title}")
            # Save the generated report to the database
            new_report = Report.objects.create(
                user=request.user,
                title=title,
                content=report_content
            )
            print("[GenerateReportView] Report saved to database successfully.")

            # Prepare the new report data for the frontend
            new_report_data = {
                'id': new_report.id,
                'title': new_report.title,
                'content': new_report.content,
                'created_at': new_report.created_at.strftime('%b %d, %Y, %I:%M %p')
            }

            return Response({
                'report': report_content,
                'new_report': new_report_data
            }, status=status.HTTP_200_OK)
        except Exception as e:
            print(f"--- [GenerateReportView] UNEXPECTED ERROR ---")
            print(f"Error in RAG pipeline for report generation: {e}")
            import traceback
            traceback.print_exc()
            print("--- END OF TRACEBACK ---")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ['message', 'is_user_message', 'created_at']


class ChatHistoryView(APIView):
    """
    API View to fetch the message history of a specific chat session.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        # Ensure the session exists and belongs to the current user
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        messages = session.messages.all()
        serializer = ChatMessageSerializer(messages, many=True)
        return Response(serializer.data)

class ReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = Report
        fields = ['id', 'title', 'content', 'created_at']

class ReportDetailView(APIView):
    """
    API View to fetch a single report.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, report_id):
        report = get_object_or_404(Report, id=report_id, user=request.user)
        serializer = ReportSerializer(report)
        return Response(serializer.data)

from django.urls import path
from .views import (
    RAGChatView, 
    chat_page, 
    ai_dashboard, 
    planning_page, 
    GenerateReportView,
    ChatHistoryView,
    ReportDetailView
)

urlpatterns = [
    path('', ai_dashboard, name='ai_dashboard'), # New AI Dashboard landing page
    path('chat/', chat_page, name='rag_chat_page'),
    path('planning/', planning_page, name='planning_page'), # New planning page
    path('api/chat/', RAGChatView.as_view(), name='rag_chat_api'),
    path('api/generate-report/', GenerateReportView.as_view(), name='generate_report'),
    path('api/chat_history/<int:session_id>/', ChatHistoryView.as_view(), name='chat_history_api'),
    path('api/report/<int:report_id>/', ReportDetailView.as_view(), name='report_detail_api'),
]

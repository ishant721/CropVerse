from django.contrib.auth import logout, authenticate, login
from django.shortcuts import render, redirect
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.views import TokenObtainPairView
from .serializers import (
    UserSerializer, VerifyOTPSerializer, LoginSerializer, ForgotPasswordSerializer,
    ResetPasswordSerializer, VerifyPasswordResetOTPSerializer, ResendOTPSerializer
)
from .models import CustomUser, Profile
from django.utils import timezone
from datetime import timedelta
import uuid
from django.core.mail import send_mail
from django.conf import settings
import random

class ResendOTPView(APIView):
    def post(self, request):
        serializer = ResendOTPSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            try:
                user = CustomUser.objects.get(email=email)
                profile = user.profile
                
                otp = str(random.randint(100000, 999999))
                profile.otp = otp
                profile.save()
                
                send_mail(
                    'Your new OTP',
                    f'Your new OTP is: {otp}',
                    settings.EMAIL_HOST_USER,
                    [email],
                    fail_silently=False,
                )
                return Response({'message': 'A new OTP has been sent to your email.'}, status=status.HTTP_200_OK)
            except CustomUser.DoesNotExist:
                return Response({'error': 'User with this email does not exist.'}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class RegisterView(generics.CreateAPIView):
    queryset = CustomUser.objects.all()
    serializer_class = UserSerializer

class VerifyOTPView(APIView):
    def post(self, request):
        serializer = VerifyOTPSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            otp = serializer.validated_data['otp']
            try:
                profile = Profile.objects.get(user__email=email)
                if profile.otp == otp:
                    profile.is_verified = True
                    profile.save()
                    return Response({'message': 'OTP verified successfully.'}, status=status.HTTP_200_OK)
                else:
                    return Response({'error': 'Invalid OTP.'}, status=status.HTTP_400_BAD_REQUEST)
            except Profile.DoesNotExist:
                return Response({'error': 'Profile not found.'}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class CustomTokenObtainPairView(TokenObtainPairView):
    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        
        try:
            serializer.is_valid(raise_exception=True)
        except Exception as e:
            return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)

        user = serializer.user
        if not user.profile.is_verified:
            # Generate and send a new OTP
            otp = str(random.randint(100000, 999999))
            user.profile.otp = otp
            user.profile.save()

            send_mail(
                'New OTP for Verification',
                f'Your new OTP is: {otp}',
                settings.EMAIL_HOST_USER,
                [user.email],
                fail_silently=False,
            )

            return Response({
                'error': 'otp_not_verified',
                'message': 'A new OTP has been sent to your email. Please verify your OTP before logging in.',
                'redirect_url': '/accounts/verify-otp/'
            }, status=status.HTTP_403_FORBIDDEN)
            
        return super().post(request, *args, **kwargs)

class ForgotPasswordView(APIView):
    def post(self, request):
        serializer = ForgotPasswordSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            try:
                user = CustomUser.objects.get(email=email)
                profile = user.profile
                
                otp = str(random.randint(100000, 999999))
                profile.otp = otp
                profile.save()
                
                send_mail(
                    'Password Reset OTP',
                    f'Your OTP to reset your password is: {otp}',
                    settings.EMAIL_HOST_USER,
                    [email],
                    fail_silently=False,
                )
                return Response({'message': 'OTP for password reset sent to your email.'}, status=status.HTTP_200_OK)
            except CustomUser.DoesNotExist:
                return Response({'error': 'User with this email does not exist.'}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class VerifyPasswordResetOTPView(APIView):
    def post(self, request):
        serializer = VerifyPasswordResetOTPSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            otp = serializer.validated_data['otp']
            try:
                profile = Profile.objects.get(user__email=email)
                if profile.otp == otp:
                    profile.otp_verified_for_password_reset = True
                    profile.save()
                    return Response({'message': 'OTP verified successfully.'}, status=status.HTTP_200_OK)
                else:
                    return Response({'error': 'Invalid OTP.'}, status=status.HTTP_400_BAD_REQUEST)
            except Profile.DoesNotExist:
                return Response({'error': 'Profile not found.'}, status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

import logging

logger = logging.getLogger(__name__)

class ResetPasswordView(APIView):
    def post(self, request):
        serializer = ResetPasswordSerializer(data=request.data)
        logger.info(f"Reset password request data: {request.data}")
        if serializer.is_valid():
            email = serializer.validated_data['email']
            password = serializer.validated_data['password']
            try:
                profile = Profile.objects.get(user__email=email)
                if not profile.otp_verified_for_password_reset:
                    return Response({'error': 'Please verify your OTP first.'}, status=status.HTTP_400_BAD_REQUEST)
                
                user = profile.user
                user.set_password(password)
                user.save()
                
                profile.otp = None
                profile.otp_verified_for_password_reset = False
                profile.save()
                
                return Response({'message': 'Password reset successfully.'}, status=status.HTTP_200_OK)
            except Profile.DoesNotExist:
                return Response({'error': 'Profile not found.'}, status=status.HTTP_404_NOT_NOT_FOUND)
        
        logger.error(f"Reset password serializer errors: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Template rendering views
def register_page(request):
    return render(request, 'accounts/register.html')

def login_page(request):
    return render(request, 'accounts/login.html')

def user_login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, email=email, password=password)
        if user is not None:
            if not user.profile.is_verified:
                # Generate and send a new OTP (optional, but good practice)
                otp = str(random.randint(100000, 999999))
                user.profile.otp = otp
                user.profile.save()

                send_mail(
                    'New OTP for Verification',
                    f'Your new OTP is: {otp}',
                    settings.EMAIL_HOST_USER,
                    [user.email],
                    fail_silently=False,
                )
                # Store email in session to retrieve on OTP verification page
                request.session['unverified_email'] = email
                return redirect('verify_otp_page')
            
            login(request, user)
            return redirect('home') # Redirect to your home page
        else:
            # Return an error message to the login page
            return render(request, 'accounts/login.html', {'error_message': 'Invalid credentials'})
    return redirect('login_page') # Should not be reached via GET, but redirect just in case

def verify_otp_page(request):
    return render(request, 'accounts/verify_otp.html')

def forgot_password_page(request):
    return render(request, 'accounts/forgot_password.html')

def reset_password_page(request):
    return render(request, 'accounts/reset_password.html')

def home_page(request):
    return render(request, 'home.html')

def logout_view(request):
    logout(request)
    return redirect('login_page') # Redirect to the named login page URL

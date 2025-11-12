from django.urls import path
from .views import (
    RegisterView, VerifyOTPView, CustomTokenObtainPairView, 
    register_page, login_page, verify_otp_page,
    ForgotPasswordView, ResetPasswordView,
    forgot_password_page, reset_password_page,
    VerifyPasswordResetOTPView, ResendOTPView,
    logout_view, user_login # Add logout_view and user_login here
)
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    path('api/register/', RegisterView.as_view(), name='register'),
    path('api/verify-otp/', VerifyOTPView.as_view(), name='verify-otp'),
    path('api/login/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/forgot-password/', ForgotPasswordView.as_view(), name='forgot_password'),
    path('api/verify-password-reset-otp/', VerifyPasswordResetOTPView.as_view(), name='verify_password_reset_otp'),
    path('api/reset-password/', ResetPasswordView.as_view(), name='reset_password'),
    path('api/resend-otp/', ResendOTPView.as_view(), name='resend_otp'),

    # Template-based views
    path('register/', register_page, name='register_page'),
    path('login/', login_page, name='login_page'),
    path('login/perform/', user_login, name='user_login'), # New URL for login form submission
    path('verify-otp/', verify_otp_page, name='verify_otp_page'),
    path('forgot-password/', forgot_password_page, name='forgot_password_page'),
    path('reset-password/', reset_password_page, name='reset_password_page'),
    path('logout/', logout_view, name='logout'), # Add logout URL
]

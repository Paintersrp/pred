"""pred_django URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path, include
from users import views as user_view
from payout import views as payout_view
from hedge import views as hedge_view
from hold import views as hold_view
from django.views.generic import TemplateView

urlpatterns = [
    path("admin/", admin.site.urls),
    # re_path(r"^api/users/$", user_view.users_list),
    # re_path(r"^api/users/([0-9])$", user_view.users_detail),
    path("api/register/", user_view.register_view),
    path("api/login/", user_view.login_view),
    path("api/user/", user_view.user_view),
    path("api/logout/", user_view.logout_view),
    re_path(r"^api/payout/$", payout_view.payout_detail),
    re_path(r"^api/hedge/$", hedge_view.hedge_detail),
    re_path(r"^api/hold/$", hold_view.hold_detail),
    re_path(r"", TemplateView.as_view(template_name="index.html")),
]

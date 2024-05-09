from models.users import User, Usercreate, UserResponse, UserLogin
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from config.database import get_db
from fastapi.security import OAuth2PasswordRequestForm
from services.Login_Service import (
    oauth2_scheme, 
    get_current_user_authorization, 
    get_user_by_login_id,
    create_user,
    authenticate_user,
    create_access_token,
    refresh_access_token,
    create_refresh_token
)

router = APIRouter()

# 유저 토큰값 조회 API
@router.get("/auth")
async def read_user_me(request: Request, token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    # 토큰을 사용하여 사용자 인증을 수행합니다.
    login_id = await get_current_user_authorization(request, token)
    user = await get_user_by_login_id(db, login_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

# ID로 사용자 읽기
@router.get("/read/{user_id}")
async def get_user_by_id(user_id: int, db: AsyncSession = Depends(get_db)):
    try:
        user = await db.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Internal Server Error", "detail": str(e)})

# 사용자 생성 API
@router.post("/signup", response_model = UserResponse)
async def signup(user: Usercreate, db: AsyncSession = Depends(get_db)):
    try:
        db_user = await create_user(db, user)       
        return db_user
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 유저 로그인 API
@router.post("/login")
async def login(user: UserLogin, db: AsyncSession = Depends(get_db)):
    db_user = await authenticate_user(db, user.login_id, user.login_pw)
    if not db_user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token({"sub": db_user.login_id})
    refresh_token = create_refresh_token(data={"sub": db_user.login_id})
    
    return {"login_id": db_user.login_id,
            "access_token": access_token,
            "refresh_token": refresh_token, 
            "token_type": "bearer"
            }

# FastAPI 내에서 테스트를 하기 위한 API
@router.post("/test")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    db_user = await authenticate_user(db, form_data.username, form_data.password)
    if not db_user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    access_token = create_access_token({"sub": db_user.login_id})
    return {"login_id" : db_user.login_id, "access_token": access_token, "token_type": "bearer"}

@router.post("/refresh")
async def refresh_token_endpoint(refresh_token: str, db: AsyncSession = Depends(get_db)):
    new_access_token = await refresh_access_token(db, refresh_token)
    return {"access_token": new_access_token, "token_type": "bearer"}
    
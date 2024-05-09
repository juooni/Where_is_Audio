from fastapi import HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta
import jwt, pytz
from jwt import PyJWTError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from models.users import User
from passlib.context import CryptContext
from config.database import SECRET_KEY, ALGORITHM

# JWT 토큰을 위한 OAuth2PasswordBearer 인스턴스 생성
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/test")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 비밀번호 해싱
def get_password_hash(password):
    return pwd_context.hash(password)

# 비밀번호 해싱 풀기
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=1)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt

def create_refresh_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=30)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# 토큰 리프레쉬
async def refresh_access_token(db: AsyncSession, refresh_token: str):
    credentials_exception = HTTPException(
        status_code=401, detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=ALGORITHM)
        login_id: str = payload.get("sub")
        if login_id is None:
            raise credentials_exception
        user = await get_user_by_login_id(db, login_id)
        if user is None:
            raise credentials_exception
        access_token = create_access_token(data={"sub": user.login_id})
        return access_token
    except jwt.PyJWTError:
        raise credentials_exception

async def get_user_by_login_id(db: AsyncSession, login_id: str):
    result = await db.execute(select(User).filter_by(login_id=login_id))
    return result.scalar_one_or_none()

# 아이디 만들기
async def create_user(db: AsyncSession, user: User):
    existing_user = await get_user_by_login_id(db, user.login_id)
    
    if existing_user:
        raise ValueError("ID Already Registered")
    
    # 한국 시간대(KST, UTC+9)를 사용하여 'created_at'을 설정합니다.
    korea_time_zone = pytz.timezone("Asia/Seoul")
    created_at_kst = datetime.now(korea_time_zone)
    
    hashed_password = get_password_hash(user.login_pw)
    
    db_user = User(
        login_id=user.login_id,
        login_pw=hashed_password,
        created_at=created_at_kst
    )
    
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)

    return db_user

async def authenticate_user(db: AsyncSession, login_id: str, login_pw: str):
    user = await get_user_by_login_id(db, login_id)
    if not user:
        raise ValueError("Invalid Credentials")
    if not verify_password(login_pw, user.login_pw):
        raise ValueError("Invalid Credentials")
    return user
    
async def get_current_user_authorization(request: Request, token : str):
    authorization: str = request.headers.get("Authorization")
    if not authorization:
        raise HTTPException(status_code=401, detail="Not Authenticated")

    token = authorization.replace("Bearer ", "")
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=ALGORITHM)
        login_id = payload.get("sub")
        if login_id is None:
            raise HTTPException(status_code=401, detail="Invalid Token")
        return login_id
    except PyJWTError as e:
        raise HTTPException(status_code=401, detail="Invalid Token")

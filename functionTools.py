''' 
    创建时间：2026-07-28
    修改时间：2026-07-28
    版本：V_0.4 - 智能体调用工具


'''


from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib
from email.header import Header
from dotenv import load_dotenv
import os
import functionTools

# 加载根目录下的 .env 文件
load_dotenv()

# 邮件服务器配置
host = os.getenv("SMTP_SERVER", "smtp.163.com")
port = os.getenv('SMTP_PORT', '25')
user = os.getenv("SENDER_EMAIL", "eastyell@163.com")
password = os.getenv("SENDER_PASSWORD", "")
cc_email = "eastyell@163.com" # 抄送邮箱

# 发送邮件
def send_email(tostr, subject, body, files_path=[]):
    if not user or not password:
            print(f"⚠️ 邮件配置未完成，模拟发送：收件人={tostr}，主题={subject}，正文={body[:50]}...")
            return f"⚠️ 邮件配置未完成，模拟发送：收件人={tostr}，主题={subject}，正文={body[:50]}..."
    try:
        msg = MIMEMultipart()
        msg['From'] = user
        msg['To'] = tostr
        msg['Cc'] = 'eastyell@163.com'  # 抄送邮箱
        # ⭐️ 修复 1：主题如果包含中文，必须使用 Header 编码
        msg['Subject'] = Header(subject, 'utf-8').encode()

        # ⭐️ 修复 2：正文必须明确指定 utf-8 编码，防止中文字符引发 str.encode 错误
        text = MIMEText(body, 'plain', 'utf-8')
        msg.attach(text) 
        # for i, image_path in enumerate(image_paths, start=1):
        #     with open(image_path, 'rb') as f:
        #         image_file = image_paths[i - 1]
        #         image_file = os.path.basename(image_file)
        #         # image_file = cut_out_str('/', image_file)
        #         image = MIMEImage(f.read())
        #         image.add_header('Content-Disposition', 'attachment', filename=image_file)
        #         msg.attach(image)
        # 添加多个附件（支持任意类型
        for file_path in files_path:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:  # 读取文件内容为二进制数据
                    # 创建一个MIMEBase对象并附加到邮件中
                    part = MIMEBase('application', 'octet-stream')  # 表示任意二进制文件
                    part.set_payload(f.read())
                    encoders.encode_base64(part)  # 对内容进行 base64 编码
                    filename = os.path.basename(file_path)
                    # ⭐️ 修复 3：附件文件名如果包含中文，必须使用 Header 编码
                    encoded_filename = Header(filename, 'utf-8').encode()
                    part.add_header(
                        'Content-Disposition',
                        'attachment',
                        filename=encoded_filename
                    )
                    msg.attach(part)
            else:
                print(f"附件文件不存在：{file_path}")
        print(f"📡 正在通过 SSL 连接 {host}:25  发送邮件 ...")
        # smtp_obj = smtplib.SMTP_SSL(host, port, timeout=15) # 增加 15 秒超时防止挂起          
        smtp_obj = smtplib.SMTP(host, port)
        smtp_obj.login(user, password)
        # smtp_obj.sendmail(msg['From'], [msg['To'], msg['Cc']], msg.as_string())
        all_receivers = [tostr, cc_email]
        smtp_obj.sendmail(user, all_receivers, msg.as_string())
        smtp_obj.quit()  
        print(f"✅ 邮件已成功发送至 {tostr}，主题：{subject}")
        return f"✅ 邮件已成功发送至 {tostr}，主题：{subject}"
    except Exception as e:
        print("❌ 发送邮件时发生异常：{}".format(e))
        return f"❌ 邮件发送失败：{str(e)}"



if __name__ == "__main__":
    send_email(user, '测试邮件','这是一封测试邮件')
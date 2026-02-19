import type React from "react";

interface ButtonProps {
  description: string;
  onClick: () => void;
  buttonType: React.ButtonHTMLAttributes<HTMLButtonElement>["type"];
}

function Button({ onClick, buttonType, description }: ButtonProps) {
  return (
    <button className="button" onClick={onClick} type={buttonType}>
      {description}
    </button>
  );
}

export default Button;

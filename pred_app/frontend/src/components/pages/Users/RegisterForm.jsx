import { useState } from "react";
import { Navigate } from "react-router-dom";
import { Link } from "react-router-dom";
import "./Users.css";

function RegisterForm() {
  const initialValues = { email: "", name: "", password: "", password2: "" };
  const [formValues, setFormValues] = useState(initialValues);
  const [formErrors, setFormErrors] = useState({});
  const [accountErrors, setAccountErrors] = useState({});
  const [redirect, setRedirect] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormValues({ ...formValues, [name]: value });
  };

  const validate = (values) => {
    const errors = {};
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/i;

    if (!values.email) {
      errors.email = "Email is required.";
    } else if (!regex.test(values.email)) {
      errors.email = "Email is invalid.";
    }

    if (!values.name) {
      errors.name = "Preferred Name is required.";
    } else if (values.name.length < 2) {
      errors.name = "Name must be at least 2 characters.";
    }

    if (!values.password) {
      errors.password = "Password is required.";
    } else if (values.password.length < 8) {
      errors.password = "Password must be at least 8 characters.";
    }

    if (!values.password2) {
      errors.password2 = "Password is required.";
    } else if (values.password2.length < 8) {
      errors.password2 = "Password must be at least 8 characters.";
    } else if (values.password2 !== values.password) {
      errors.password = "Password entries do not match.";
      errors.password2 = "Password entries do not match.";
    }

    return errors;
  };

  const validateResponse = (values) => {
    const errors = {};

    if (values.email) {
      console.log(values.email);
      errors.email = "Accounts already exists for email address.";
    }

    return errors;
  };

  const submit = async (e) => {
    e.preventDefault();
    setFormErrors(validate(formValues));

    if (Object.keys(formErrors).length === 0) {
      const response = await fetch("http://54.161.55.120:8000/api/register/", {
        method: "POST",
        // headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          formValues,
        }),
      });

      const content = await response.json();
      console.log(response.status);

      setAccountErrors(validateResponse(content));

      if (response.status === 200) {
        setRedirect(true);
      } else if (response.status === 201) {
        setRedirect(true);
      }
    }
  };

  if (redirect) {
    return <Navigate to="/login" />;
  }

  return (
    <div className="layout-form flex-wrap">
      <div className="w-[400px] form-container">
        <form onSubmit={submit} className="mb-12">
          <h2 className="mb-4 mt-10 text-center text-2xl font-[500] underline underline-offset-8">
            Register Now
          </h2>
          <p className="mb-4 error-msg">{accountErrors.email}</p>
          <div className="flex flex-col justify-center text-center">
            <label htmlFor="email" className="mb-1">
              Email
            </label>
            <input
              className="form-input text-[14px] w-[300px]"
              type="email"
              name="email"
              value={formValues.email}
              placeholder="Enter email"
              onChange={handleChange}
            />
          </div>
          <p className="mt-1 error-msg">{formErrors.email}</p>
          <div className="flex flex-col justify-center text-center">
            <label htmlFor="name" className="mt-4 mb-1">
              Preferred Name
            </label>
            <input
              className="form-input text-[14px] w-[300px]"
              type="text"
              name="name"
              value={formValues.name}
              placeholder="Enter preferred name"
              onChange={handleChange}
            />
          </div>
          <p className="mt-1 error-msg">{formErrors.name}</p>
          <div className="flex flex-col justify-center text-center">
            <label htmlFor="password" className="mt-4 mb-1">
              Password
            </label>
            <input
              className="form-input text-[14px] w-[300px]"
              type="password"
              name="password"
              value={formValues.password}
              placeholder="Enter password"
              onChange={handleChange}
            />
          </div>
          <p className="mt-1 error-msg">{formErrors.password}</p>
          <div className="flex flex-col justify-center text-center">
            <label htmlFor="password2" className="mt-4 mb-1">
              Verify Password
            </label>
            <input
              className="form-input text-[14px] w-[300px]"
              type="password"
              name="password2"
              value={formValues.password2}
              placeholder="Verify password"
              onChange={handleChange}
            />
          </div>
          <p className="mt-1 error-msg">{formErrors.password2}</p>
          <div className="flex justify-center">
            <button className="w-[50%] mt-6 form-btn font-[500]">
              Register
            </button>
          </div>
          <span className="mt-4 text-[14px] flex justify-center">
            Already have an account?{" "}
            <Link
              to="/login"
              className="ml-1 underline underline-offset-4 link-color"
            >
              Login here
            </Link>
          </span>
        </form>
      </div>
    </div>
  );
}

export default RegisterForm;

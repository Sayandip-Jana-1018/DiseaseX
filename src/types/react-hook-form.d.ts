declare module 'react-hook-form' {
  import * as React from 'react';
  
  export type FieldValues = Record<string, any>;
  
  export type RegisterOptions = {
    required?: boolean | string;
    min?: { value: number; message: string };
    max?: { value: number; message: string };
    maxLength?: { value: number; message: string };
    minLength?: { value: number; message: string };
    pattern?: { value: RegExp; message: string };
    validate?: (value: any) => boolean | string | Promise<boolean | string>;
    onChange?: (event: any) => void;
  };
  
  export type FieldError = {
    type: string;
    message?: string;
  };
  
  export type FieldErrors<T extends FieldValues = FieldValues> = {
    [K in keyof T]?: FieldError;
  };
  
  export type UseFormRegister<T extends FieldValues = FieldValues> = (name: keyof T, options?: RegisterOptions) => {
    name: string;
    onChange: (...event: any[]) => void;
    onBlur: (...event: any[]) => void;
    ref: (ref: any) => void;
  };
  
  export type UseFormHandleSubmit<T extends FieldValues = FieldValues> = (onSubmit: (data: T) => void) => (e: React.FormEvent) => void;
  
  export type FormState<T extends FieldValues = FieldValues> = {
    isDirty: boolean;
    isSubmitting: boolean;
    isSubmitted: boolean;
    isSubmitSuccessful: boolean;
    isValid: boolean;
    dirtyFields: Record<keyof T, boolean>;
    touchedFields: Record<keyof T, boolean>;
    errors: FieldErrors<T>;
  };
  
  export type Control<T extends FieldValues = FieldValues> = {
    _defaultValues: Partial<T>;
    _formState: FormState<T>;
    _options: any;
  };
  
  export interface ControllerProps<T extends FieldValues = FieldValues> {
    name: keyof T;
    control?: Control<T>;
    defaultValue?: any;
    rules?: RegisterOptions;
    render: ({ field, fieldState }: {
      field: {
        onChange: (...event: any[]) => void;
        onBlur: () => void;
        value: any;
        name: string;
        ref: React.Ref<any>;
      };
      fieldState: {
        invalid: boolean;
        isTouched: boolean;
        isDirty: boolean;
        error?: FieldError;
      };
    }) => React.ReactElement;
  }
  
  export const Controller: React.FC<ControllerProps>;
  
  export function useForm<T extends FieldValues = FieldValues>(options?: {
    mode?: 'onSubmit' | 'onChange' | 'onBlur' | 'onTouched' | 'all';
    reValidateMode?: 'onSubmit' | 'onChange' | 'onBlur';
    defaultValues?: Partial<T>;
    resolver?: any;
    context?: any;
    criteriaMode?: 'firstError' | 'all';
    shouldFocusError?: boolean;
    shouldUnregister?: boolean;
  }): {
    register: UseFormRegister<T>;
    handleSubmit: UseFormHandleSubmit<T>;
    formState: FormState<T>;
    watch: (name?: string | string[]) => any;
    setValue: (name: keyof T, value: any, options?: { shouldValidate?: boolean; shouldDirty?: boolean }) => void;
    getValues: (payload?: string | string[]) => any;
    reset: (values?: Partial<T>, options?: { keepErrors?: boolean; keepDirty?: boolean; keepIsSubmitted?: boolean; keepTouched?: boolean; keepIsValid?: boolean; keepSubmitCount?: boolean }) => void;
    clearErrors: (name?: keyof T | (keyof T)[]) => void;
    setError: (name: keyof T, error: FieldError) => void;
    trigger: (name?: keyof T | (keyof T)[]) => Promise<boolean>;
    control: Control<T>;
  };
}

declare module 'react-select' {
  import * as React from 'react';
  
  export interface OptionType {
    label: string;
    value: string | number;
    [key: string]: any;
  }
  
  export interface GroupType {
    label: string;
    options: OptionType[];
    [key: string]: any;
  }
  
  export interface StylesConfig {
    clearIndicator?: (provided: any, state: any) => any;
    container?: (provided: any, state: any) => any;
    control?: (provided: any, state: any) => any;
    dropdownIndicator?: (provided: any, state: any) => any;
    group?: (provided: any, state: any) => any;
    groupHeading?: (provided: any, state: any) => any;
    indicatorsContainer?: (provided: any, state: any) => any;
    indicatorSeparator?: (provided: any, state: any) => any;
    input?: (provided: any, state: any) => any;
    loadingIndicator?: (provided: any, state: any) => any;
    loadingMessage?: (provided: any, state: any) => any;
    menu?: (provided: any, state: any) => any;
    menuList?: (provided: any, state: any) => any;
    menuPortal?: (provided: any, state: any) => any;
    multiValue?: (provided: any, state: any) => any;
    multiValueLabel?: (provided: any, state: any) => any;
    multiValueRemove?: (provided: any, state: any) => any;
    noOptionsMessage?: (provided: any, state: any) => any;
    option?: (provided: any, state: any) => any;
    placeholder?: (provided: any, state: any) => any;
    singleValue?: (provided: any, state: any) => any;
    valueContainer?: (provided: any, state: any) => any;
  }
  
  export interface Props {
    className?: string;
    classNamePrefix?: string;
    isMulti?: boolean;
    isSearchable?: boolean;
    isClearable?: boolean;
    isDisabled?: boolean;
    isLoading?: boolean;
    placeholder?: string;
    options?: OptionType[] | GroupType[];
    value?: OptionType | OptionType[] | null;
    defaultValue?: OptionType | OptionType[] | null;
    onChange?: (value: OptionType | OptionType[] | null, action: any) => void;
    onInputChange?: (inputValue: string, action: any) => void;
    onMenuOpen?: () => void;
    onMenuClose?: () => void;
    onBlur?: (event: React.FocusEvent) => void;
    onFocus?: (event: React.FocusEvent) => void;
    styles?: StylesConfig;
    theme?: any;
    components?: any;
    menuPlacement?: 'auto' | 'bottom' | 'top';
    menuPosition?: 'absolute' | 'fixed';
    menuPortalTarget?: HTMLElement;
    noOptionsMessage?: (obj: { inputValue: string }) => React.ReactNode;
    formatGroupLabel?: (group: GroupType) => React.ReactNode;
    formatOptionLabel?: (option: OptionType, formatOptionLabelMeta: any) => React.ReactNode;
    getOptionLabel?: (option: OptionType) => string;
    getOptionValue?: (option: OptionType) => string;
    isOptionDisabled?: (option: OptionType) => boolean;
    closeMenuOnSelect?: boolean;
    closeMenuOnScroll?: boolean | ((e: Event) => boolean);
    hideSelectedOptions?: boolean;
    inputValue?: string;
    maxMenuHeight?: number;
    minMenuHeight?: number;
    name?: string;
    tabIndex?: number;
    tabSelectsValue?: boolean;
    autoFocus?: boolean;
    blurInputOnSelect?: boolean;
    captureMenuScroll?: boolean;
    controlShouldRenderValue?: boolean;
    escapeClearsValue?: boolean;
    filterOption?: ((option: OptionType, rawInput: string) => boolean) | null;
    formatCreateLabel?: (inputValue: string) => React.ReactNode;
    isValidNewOption?: (inputValue: string, selectValue: OptionType[], selectOptions: OptionType[]) => boolean;
    loadingMessage?: (obj: { inputValue: string }) => React.ReactNode;
    openMenuOnFocus?: boolean;
    openMenuOnClick?: boolean;
    pageSize?: number;
    screenReaderStatus?: (obj: { count: number }) => string;
    backspaceRemovesValue?: boolean;
    id?: string;
  }
  
  export default class Select extends React.Component<Props> {}
  
  export class Creatable extends React.Component<Props & {
    allowCreateWhileLoading?: boolean;
    createOptionPosition?: 'first' | 'last';
    formatCreateLabel?: (inputValue: string) => React.ReactNode;
    isValidNewOption?: (inputValue: string, selectValue: OptionType[], selectOptions: OptionType[]) => boolean;
    getNewOptionData?: (inputValue: string, optionLabel: React.ReactNode) => OptionType;
    onCreateOption?: (inputValue: string) => void;
  }> {}
  
  export class Async extends React.Component<Props & {
    cacheOptions?: boolean;
    defaultOptions?: boolean | OptionType[];
    loadOptions?: (inputValue: string, callback: (options: OptionType[]) => void) => void | Promise<OptionType[]>;
  }> {}
  
  export class AsyncCreatable extends React.Component<Props & {
    allowCreateWhileLoading?: boolean;
    createOptionPosition?: 'first' | 'last';
    formatCreateLabel?: (inputValue: string) => React.ReactNode;
    isValidNewOption?: (inputValue: string, selectValue: OptionType[], selectOptions: OptionType[]) => boolean;
    getNewOptionData?: (inputValue: string, optionLabel: React.ReactNode) => OptionType;
    onCreateOption?: (inputValue: string) => void;
    cacheOptions?: boolean;
    defaultOptions?: boolean | OptionType[];
    loadOptions?: (inputValue: string, callback: (options: OptionType[]) => void) => void | Promise<OptionType[]>;
  }> {}
  
  export const components: {
    ClearIndicator: React.ComponentType<any>;
    Control: React.ComponentType<any>;
    DropdownIndicator: React.ComponentType<any>;
    DownChevron: React.ComponentType<any>;
    CrossIcon: React.ComponentType<any>;
    Group: React.ComponentType<any>;
    GroupHeading: React.ComponentType<any>;
    IndicatorsContainer: React.ComponentType<any>;
    IndicatorSeparator: React.ComponentType<any>;
    Input: React.ComponentType<any>;
    LoadingIndicator: React.ComponentType<any>;
    Menu: React.ComponentType<any>;
    MenuList: React.ComponentType<any>;
    MenuPortal: React.ComponentType<any>;
    LoadingMessage: React.ComponentType<any>;
    NoOptionsMessage: React.ComponentType<any>;
    MultiValue: React.ComponentType<any>;
    MultiValueContainer: React.ComponentType<any>;
    MultiValueLabel: React.ComponentType<any>;
    MultiValueRemove: React.ComponentType<any>;
    Option: React.ComponentType<any>;
    Placeholder: React.ComponentType<any>;
    SelectContainer: React.ComponentType<any>;
    SingleValue: React.ComponentType<any>;
    ValueContainer: React.ComponentType<any>;
  };
}

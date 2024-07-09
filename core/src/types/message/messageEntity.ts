import { ChatCompletionMessage, ChatCompletionRole } from '../inference'
import { ModelInfo } from '../model'
import { Thread } from '../thread'

/**
 * The `ThreadMessage` type defines the shape of a thread's message object.
 * @stored
 */
export type ThreadMessage = {
  /** Unique identifier for the message, generated by default using the ULID method. **/
  id: string
  /** Object name **/
  object: string
  /** Thread id, default is a ulid. **/
  thread_id: string
  /** The assistant id of this thread. **/
  assistant_id?: string
  /** The role of the author of this message. **/
  role: ChatCompletionRole
  /** The content of this message. **/
  content: ThreadContent[]
  /** The status of this message. **/
  status: MessageStatus
  /** The timestamp indicating when this message was created. Represented in Unix time. **/
  created: number
  /** The timestamp indicating when this message was updated. Represented in Unix time. **/
  updated: number
  /** The additional metadata of this message. **/
  metadata?: Record<string, unknown>

  type?: string

  /** The error code which explain what error type. Used in conjunction with MessageStatus.Error */
  error_code?: ErrorCode
}

/**
 * The `MessageRequest` type defines the shape of a new message request object.
 * @data_transfer_object
 */
export type MessageRequest = {
  id?: string

  /**
   * @deprecated Use thread object instead
   * The thread id of the message request.
   */
  threadId: string

  /**
   * The assistant id of the message request.
   */
  assistantId?: string

  /** Messages for constructing a chat completion request **/
  messages?: ChatCompletionMessage[]

  /** Settings for constructing a chat completion request **/
  model?: ModelInfo

  /** The thread of this message is belong to. **/
  // TODO: deprecate threadId field
  thread?: Thread

  type?: string
}

/**
 * The status of the message.
 * @data_transfer_object
 */
export enum MessageStatus {
  /** Message is fully loaded. **/
  Ready = 'ready',
  /** Message is not fully loaded. **/
  Pending = 'pending',
  /** Message loaded with error. **/
  Error = 'error',
  /** Message is cancelled streaming */
  Stopped = 'stopped',
}

export enum ErrorCode {
  InvalidApiKey = 'invalid_api_key',

  AuthenticationError = 'authentication_error',

  InsufficientQuota = 'insufficient_quota',

  InvalidRequestError = 'invalid_request_error',

  Unknown = 'unknown',
}

/**
 * The content type of the message.
 */
export enum ContentType {
  Text = 'text',
  Image = 'image',
  Pdf = 'pdf',
}

/**
 * The `ContentValue` type defines the shape of a content value object
 * @data_transfer_object
 */
export type ContentValue = {
  value: string
  annotations: string[]
  name?: string
  size?: number
}

/**
 * The `ThreadContent` type defines the shape of a message's content object
 * @data_transfer_object
 */
export type ThreadContent = {
  type: ContentType
  text: ContentValue
}

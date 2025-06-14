from engine.human.agent.db import Base, get_db0, add_conversation_to_db, create_tables, reset_tables, add_message_to_db, \
    update_message, get_message_by_id, filter_message

if __name__ == '__main__':
    create_tables()
    reset_tables()

    cid = add_conversation_to_db(
       'test', 'zjh'
    )
    print(cid)

    mid = add_message_to_db(
       cid, 'test', '1+1=?', "=2"
    )
    print(mid)

    mid = update_message(
       mid, "=3"
    )
    print(mid)

    msg = get_message_by_id(
       mid, in_context=False
    )
    print(msg)

    mid = add_message_to_db(
       cid, 'test', '1+2=?', "=6"
    )
    msgs = filter_message(cid, limit=1)
    print(msgs)



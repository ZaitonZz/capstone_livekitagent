import unittest
from unittest.mock import Mock

from livekit import rtc
from tests._agent_import_stubs import install_agent_dependency_stubs

install_agent_dependency_stubs()

from agent import has_active_track_handler, list_remote_participants, list_video_tracks_for_participant


class FakeTrack:
    def __init__(self, sid: str, kind: rtc.TrackKind) -> None:
        self.sid = sid
        self.kind = kind


class FakePublication:
    def __init__(self, sid: str, kind: rtc.TrackKind, track: FakeTrack | None = None) -> None:
        self.sid = sid
        self.kind = kind
        self.track = track


class FakeParticipant:
    def __init__(
        self,
        video_tracks: dict[str, FakePublication] | None = None,
        track_publications: dict[str, FakePublication] | None = None,
        tracks: list[FakePublication] | None = None,
    ) -> None:
        self.video_tracks = video_tracks or {}
        self.track_publications = track_publications or {}
        self.tracks = tracks or []


class AgentRejoinHelpersTest(unittest.TestCase):
    def test_list_remote_participants_prefers_remote_participants(self) -> None:
        preferred_participant = object()
        fallback_participant = object()
        room = type(
            "FakeRoom",
            (),
            {
                "remote_participants": {"preferred": preferred_participant},
                "participants": {"fallback": fallback_participant},
            },
        )()

        participants = list_remote_participants(room)

        self.assertEqual(participants, [preferred_participant])

    def test_list_remote_participants_uses_participants_fallback(self) -> None:
        fallback_participant = object()
        room = type(
            "FakeRoom",
            (),
            {
                "participants": {"fallback": fallback_participant},
            },
        )()

        participants = list_remote_participants(room)

        self.assertEqual(participants, [fallback_participant])

    def test_list_video_tracks_for_participant_dedupes_track_sid(self) -> None:
        duplicated_track = FakeTrack("TR_same", rtc.TrackKind.KIND_VIDEO)
        unique_track = FakeTrack("TR_unique", rtc.TrackKind.KIND_VIDEO)
        audio_track = FakeTrack("TR_audio", rtc.TrackKind.KIND_AUDIO)

        participant = FakeParticipant(
            video_tracks={
                "pub_a": FakePublication("pub_a", rtc.TrackKind.KIND_VIDEO, duplicated_track),
            },
            track_publications={
                "pub_b": FakePublication("pub_b", rtc.TrackKind.KIND_VIDEO, duplicated_track),
                "pub_c": FakePublication("pub_c", rtc.TrackKind.KIND_VIDEO, unique_track),
                "pub_d": FakePublication("pub_d", rtc.TrackKind.KIND_AUDIO, audio_track),
            },
        )

        tracks = list_video_tracks_for_participant(participant)
        track_sids = [track.sid for track in tracks]

        self.assertEqual(track_sids, ["TR_same", "TR_unique"])

    def test_has_active_track_handler_checks_task_state(self) -> None:
        running_task = Mock()
        running_task.done.return_value = False
        finished_task = Mock()
        finished_task.done.return_value = True

        self.assertEqual(has_active_track_handler({"TR_1": running_task}, "TR_1"), True)
        self.assertEqual(has_active_track_handler({"TR_2": finished_task}, "TR_2"), False)
        self.assertEqual(has_active_track_handler({}, "TR_3"), False)


if __name__ == "__main__":
    unittest.main()

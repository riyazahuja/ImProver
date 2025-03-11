/-- The birthday of a pre-game is inductively defined as the least strict upper bound of the
birthdays of its left and right games. It may be thought as the "step" in which a certain game is
constructed. -/
noncomputable def birthday : PGame.{u} → Ordinal.{u}
  | ⟨_, _, xL, xR⟩ =>
    max (lsub.{u, u} fun i => birthday (xL i)) (lsub.{u, u} fun i => birthday (xR i))


theorem birthday_def (x : PGame) :
    birthday x =
      max (lsub.{u, u} fun i => birthday (x.moveLeft i))
        (lsub.{u, u} fun i => birthday (x.moveRight i)) := by
  /-
    x : SetTheory.PGame
    ⊢ Eq x.birthday (Max.max (Ordinal.lsub fun i => (x.moveLeft i).birthday) (Ordi …
  -/
  cases x; rw [birthday]; rfl
                          /-
                            🎉 no goals
                          -/


theorem birthday_moveLeft_lt {x : PGame} (i : x.LeftMoves) :
    (x.moveLeft i).birthday < x.birthday := by
  /-
    x : SetTheory.PGame
    i : x.LeftMoves
    ⊢ LT.lt (x.moveLeft i).birthday x.birthday
  -/
  cases x; rw [birthday]; exact lt_max_of_lt_left (lt_lsub _ i)
                          /-
                            🎉 no goals
                          -/


theorem birthday_moveRight_lt {x : PGame} (i : x.RightMoves) :
    (x.moveRight i).birthday < x.birthday := by
  /-
    x : SetTheory.PGame
    i : x.RightMoves
    ⊢ LT.lt (x.moveRight i).birthday x.birthday
  -/
  cases x; rw [birthday]; exact lt_max_of_lt_right (lt_lsub _ i)
                          /-
                            🎉 no goals
                          -/


theorem lt_birthday_iff {x : PGame} {o : Ordinal} :
    o < x.birthday ↔
      (∃ i : x.LeftMoves, o ≤ (x.moveLeft i).birthday) ∨
        ∃ i : x.RightMoves, o ≤ (x.moveRight i).birthday := by
  /-
    x : SetTheory.PGame
    o : Ordinal.{u_1}
    ⊢ Iff (LT.lt o x.birthday) (Or (Exists fun i => LE.le o (x.moveLeft i).birthda …
  -/
  constructor
    /-
      case mp
      x : SetTheory.PGame
      o : Ordinal.{u_1}
      ⊢ LT.lt o x.birthday → Or (Exists fun i => LE.le o (x.moveLeft i).birthday) (E …
    -/
  · rw [birthday_def]
    /-
      case mp
      x : SetTheory.PGame
      o : Ordinal.{u_1}
      ⊢ LT.lt o (Max.max (Ordinal.lsub fun i => (x.moveLeft i).birthday) (Ordinal.ls …
    -/
    intro h
    /-
      case mp
      x : SetTheory.PGame
      o : Ordinal.{u_1}
      h : LT.lt o (Max.max (Ordinal.lsub fun i => (x.moveLeft i).birthday) (Ordinal. …
      ⊢ Or (Exists fun i => LE.le o (x.moveLeft i).birthday) (Exists fun i => LE.le  …
    -/
    cases' lt_max_iff.1 h with h' h'
      /-
        case mp.inl
        x : SetTheory.PGame
        o : Ordinal.{u_1}
        h : LT.lt o (Max.max (Ordinal.lsub fun i => (x.moveLeft i).birthday) (Ordinal. …
        h' : LT.lt o (Ordinal.lsub fun i => (x.moveLeft i).birthday)
        ⊢ Or (Exists fun i => LE.le o (x.moveLeft i).birthday) (Exists fun i => LE.le  …
      -/
    · left
      /-
        case mp.inl.h
        x : SetTheory.PGame
        o : Ordinal.{u_1}
        h : LT.lt o (Max.max (Ordinal.lsub fun i => (x.moveLeft i).birthday) (Ordinal. …
        h' : LT.lt o (Ordinal.lsub fun i => (x.moveLeft i).birthday)
        ⊢ Exists fun i => LE.le o (x.moveLeft i).birthday
      -/
      rwa [lt_lsub_iff] at h'
      /-
        🎉 no goals
      -/
      /-
        case mp.inr
        x : SetTheory.PGame
        o : Ordinal.{u_1}
        h : LT.lt o (Max.max (Ordinal.lsub fun i => (x.moveLeft i).birthday) (Ordinal. …
        h' : LT.lt o (Ordinal.lsub fun i => (x.moveRight i).birthday)
        ⊢ Or (Exists fun i => LE.le o (x.moveLeft i).birthday) (Exists fun i => LE.le  …
      -/
    · right
      /-
        case mp.inr.h
        x : SetTheory.PGame
        o : Ordinal.{u_1}
        h : LT.lt o (Max.max (Ordinal.lsub fun i => (x.moveLeft i).birthday) (Ordinal. …
        h' : LT.lt o (Ordinal.lsub fun i => (x.moveRight i).birthday)
        ⊢ Exists fun i => LE.le o (x.moveRight i).birthday
      -/
      rwa [lt_lsub_iff] at h'
      /-
        🎉 no goals
      -/
    /-
      case mpr
      x : SetTheory.PGame
      o : Ordinal.{u_1}
      ⊢ Or (Exists fun i => LE.le o (x.moveLeft i).birthday) (Exists fun i => LE.le  …
    -/
  · rintro (⟨i, hi⟩ | ⟨i, hi⟩)
      /-
        case mpr.inl.intro
        x : SetTheory.PGame
        o : Ordinal.{u_1}
        i : x.LeftMoves
        hi : LE.le o (x.moveLeft i).birthday
        ⊢ LT.lt o x.birthday
      -/
    · exact hi.trans_lt (birthday_moveLeft_lt i)
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.intro
        x : SetTheory.PGame
        o : Ordinal.{u_1}
        i : x.RightMoves
        hi : LE.le o (x.moveRight i).birthday
        ⊢ LT.lt o x.birthday
      -/
    · exact hi.trans_lt (birthday_moveRight_lt i)
      /-
        🎉 no goals
      -/


theorem Relabelling.birthday_congr : ∀ {x y : PGame.{u}}, x ≡r y → birthday x = birthday y
  | ⟨xl, xr, xL, xR⟩, ⟨yl, yr, yL, yR⟩, r => by
    /-
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      r : (SetTheory.PGame.mk xl xr xL xR).Relabelling (SetTheory.PGame.mk yl yr yL  …
      ⊢ Eq (SetTheory.PGame.mk xl xr xL xR).birthday (SetTheory.PGame.mk yl yr yL yR …
    -/
    unfold birthday
    /-
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      r : (SetTheory.PGame.mk xl xr xL xR).Relabelling (SetTheory.PGame.mk yl yr yL  …
      ⊢ Eq (Max.max (Ordinal.lsub fun i => (xL i).birthday) (Ordinal.lsub fun i => ( …
    -/
    congr 1
    all_goals
      apply lsub_eq_of_range_eq.{u, u, u}
      ext i; constructor
    /-
      case e_a.h.mp
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      r : (SetTheory.PGame.mk xl xr xL xR).Relabelling (SetTheory.PGame.mk yl yr yL  …
      i : Ordinal.{u}
      ⊢ Membership.mem (Set.range fun i => (xL i).birthday) i → Membership.mem (Set. …
    -/
    all_goals rintro ⟨j, rfl⟩
      /-
        case e_a.h.mp.intro
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        r : (SetTheory.PGame.mk xl xr xL xR).Relabelling (SetTheory.PGame.mk yl yr yL  …
        j : xl
        ⊢ Membership.mem (Set.range fun i => (yL i).birthday) ((fun i => (xL i).birthd …
      -/
    · exact ⟨_, (r.moveLeft j).birthday_congr.symm⟩
      /-
        🎉 no goals
      -/
      /-
        case e_a.h.mpr.intro
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        r : (SetTheory.PGame.mk xl xr xL xR).Relabelling (SetTheory.PGame.mk yl yr yL  …
        j : yl
        ⊢ Membership.mem (Set.range fun i => (xL i).birthday) ((fun i => (yL i).birthd …
      -/
    · exact ⟨_, (r.moveLeftSymm j).birthday_congr⟩
      /-
        🎉 no goals
      -/
      /-
        case e_a.h.mp.intro
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        r : (SetTheory.PGame.mk xl xr xL xR).Relabelling (SetTheory.PGame.mk yl yr yL  …
        j : xr
        ⊢ Membership.mem (Set.range fun i => (yR i).birthday) ((fun i => (xR i).birthd …
      -/
    · exact ⟨_, (r.moveRight j).birthday_congr.symm⟩
      /-
        🎉 no goals
      -/
      /-
        case e_a.h.mpr.intro
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        r : (SetTheory.PGame.mk xl xr xL xR).Relabelling (SetTheory.PGame.mk yl yr yL  …
        j : yr
        ⊢ Membership.mem (Set.range fun i => (xR i).birthday) ((fun i => (yR i).birthd …
      -/
    · exact ⟨_, (r.moveRightSymm j).birthday_congr⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem birthday_eq_zero {x : PGame} :
    birthday x = 0 ↔ IsEmpty x.LeftMoves ∧ IsEmpty x.RightMoves := by
  /-
    x : SetTheory.PGame
    ⊢ Iff (Eq x.birthday 0) (And (IsEmpty x.LeftMoves) (IsEmpty x.RightMoves))
  -/
  rw [birthday_def, max_eq_zero, lsub_eq_zero_iff, lsub_eq_zero_iff]
  /-
    🎉 no goals
  -/


@[simp]
                                             /-
                                               ⊢ Eq (SetTheory.PGame.birthday 0) 0
                                             -/
theorem birthday_zero : birthday 0 = 0 := by simp [inferInstanceAs (IsEmpty PEmpty)]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
                                            /-
                                              ⊢ Eq (SetTheory.PGame.birthday 1) 1
                                            -/
theorem birthday_one : birthday 1 = 1 := by rw [birthday_def]; simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
                                                /-
                                                  ⊢ Eq SetTheory.PGame.star.birthday 1
                                                -/
theorem birthday_star : birthday star = 1 := by rw [birthday_def]; simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem birthday_neg : ∀ x : PGame, (-x).birthday = x.birthday
  | ⟨xl, xr, xL, xR⟩ => by
    /-
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      ⊢ Eq (Neg.neg (SetTheory.PGame.mk xl xr xL xR)).birthday (SetTheory.PGame.mk x …
    -/
    rw [birthday_def, birthday_def, max_comm]
    /-
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      ⊢ Eq (Max.max (Ordinal.lsub fun i => ((Neg.neg (SetTheory.PGame.mk xl xr xL xR …
    -/
                         /-
                           🎉 no goals
                         -/
    congr <;> funext <;> apply birthday_neg
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem birthday_ordinalToPGame (o : Ordinal) : o.toPGame.birthday = o := by
  /-
    o : Ordinal.{u_1}
    ⊢ Eq o.toPGame.birthday o
  -/
  induction' o using Ordinal.induction with o IH
  /-
    case h
    o : Ordinal.{u_1}
    IH : ∀ (k : Ordinal.{u_1}), LT.lt k o → Eq k.toPGame.birthday k
    ⊢ Eq o.toPGame.birthday o
  -/
  rw [toPGame, PGame.birthday]
  /-
    case h
    o : Ordinal.{u_1}
    IH : ∀ (k : Ordinal.{u_1}), LT.lt k o → Eq k.toPGame.birthday k
    ⊢ Eq (Max.max (Ordinal.lsub fun i => (↑(o.enumIsoToType.symm i)).toPGame.birth …
  -/
  simp only [lsub_empty, max_zero_right]
  /-
    case h
    o : Ordinal.{u_1}
    IH : ∀ (k : Ordinal.{u_1}), LT.lt k o → Eq k.toPGame.birthday k
    ⊢ Eq (Ordinal.lsub fun i => (↑(o.enumIsoToType.symm i)).toPGame.birthday) o
  -/
  conv_rhs => rw [← lsub_typein o]
  /-
    case h
    o : Ordinal.{u_1}
    IH : ∀ (k : Ordinal.{u_1}), LT.lt k o → Eq k.toPGame.birthday k
    ⊢ Eq (Ordinal.lsub fun i => (↑(o.enumIsoToType.symm i)).toPGame.birthday) (Ord …
  -/
  congr with x
  /-
    case h.e_f.h
    o : Ordinal.{u_1}
    IH : ∀ (k : Ordinal.{u_1}), LT.lt k o → Eq k.toPGame.birthday k
    x : o.toType
    ⊢ Eq (↑(o.enumIsoToType.symm x)).toPGame.birthday ((Ordinal.typein fun x1 x2 = …
  -/
  exact IH _ (typein_lt_self x)
  /-
    🎉 no goals
  -/


theorem le_birthday : ∀ x : PGame, x ≤ x.birthday.toPGame
  | ⟨xl, _, xL, _⟩ =>
    le_def.2
      ⟨fun i =>
                                                                   /-
                                                                     xl β✝ : Type u_1
                                                                     xL : xl → SetTheory.PGame
                                                                     a✝ : β✝ → SetTheory.PGame
                                                                     i : (SetTheory.PGame.mk xl β✝ xL a✝).LeftMoves
                                                                     ⊢ LE.le ((SetTheory.PGame.mk xl β✝ xL a✝).moveLeft i) ((SetTheory.PGame.mk xl  …
                                                                   -/
        Or.inl ⟨toLeftMovesToPGame ⟨_, birthday_moveLeft_lt i⟩, by simp [le_birthday (xL i)]⟩,
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
        isEmptyElim⟩


theorem neg_birthday_le : -x.birthday.toPGame ≤ x := by
  /-
    x : SetTheory.PGame
    ⊢ LE.le (Neg.neg x.birthday.toPGame) x
  -/
  simpa only [birthday_neg, ← neg_le_iff] using le_birthday (-x)
  /-
    🎉 no goals
  -/


@[simp]
theorem birthday_add : ∀ x y : PGame, (x + y).birthday = x.birthday ♯ y.birthday
  | ⟨xl, xr, xL, xR⟩, ⟨yl, yr, yL, yR⟩ => by
    /-
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      ⊢ Eq (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk yl yr yL  …
    -/
    rw [birthday_def, nadd_def, lsub_sum, lsub_sum]
    simp only [mk_add_moveLeft_inl, mk_add_moveLeft_inr, mk_add_moveRight_inl, mk_add_moveRight_inr,
      moveLeft_mk, moveRight_mk]
    -- Porting note: Originally `simp only [birthday_add]`, but this causes an error in
    -- `termination_by`. Use a workaround.
    /-
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      ⊢ Eq (Max.max (Max.max (Ordinal.lsub fun a => (HAdd.hAdd (xL a) (SetTheory.PGa …
    -/
    conv_lhs => left; left; right; intro a; rw [birthday_add (xL a) ⟨yl, yr, yL, yR⟩]
    /-
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      ⊢ Eq (Max.max (Max.max (Ordinal.lsub fun a => (xL a).birthday.nadd (SetTheory. …
    -/
    conv_lhs => left; right; right; intro b; rw [birthday_add ⟨xl, xr, xL, xR⟩ (yL b)]
    /-
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      ⊢ Eq (Max.max (Max.max (Ordinal.lsub fun a => (xL a).birthday.nadd (SetTheory. …
    -/
    conv_lhs => right; left; right; intro a; rw [birthday_add (xR a) ⟨yl, yr, yL, yR⟩]
    /-
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      ⊢ Eq (Max.max (Max.max (Ordinal.lsub fun a => (xL a).birthday.nadd (SetTheory. …
    -/
    conv_lhs => right; right; right; intro b; rw [birthday_add ⟨xl, xr, xL, xR⟩ (yR b)]
    /-
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      ⊢ Eq (Max.max (Max.max (Ordinal.lsub fun a => (xL a).birthday.nadd (SetTheory. …
    -/
    rw [max_max_max_comm]
    /-
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      ⊢ Eq (Max.max (Max.max (Ordinal.lsub fun a => (xL a).birthday.nadd (SetTheory. …
    -/
    congr <;> apply le_antisymm
    any_goals
      exact
        max_le_iff.2
          ⟨lsub_le_iff.2 fun i => lt_blsub _ _ (birthday_moveLeft_lt _),
            lsub_le_iff.2 fun i => lt_blsub _ _ (birthday_moveRight_lt _)⟩
    all_goals
      refine blsub_le_iff.2 fun i hi => ?_
      rcases lt_birthday_iff.1 hi with (⟨j, hj⟩ | ⟨j, hj⟩)
      /-
        case e_a.a.inl.intro
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        i : Ordinal.{u_1}
        hi : LT.lt i (SetTheory.PGame.mk xl xr xL xR).birthday
        j : (SetTheory.PGame.mk xl xr xL xR).LeftMoves
        hj : LE.le i ((SetTheory.PGame.mk xl xr xL xR).moveLeft j).birthday
        ⊢ LT.lt (i.nadd (SetTheory.PGame.mk yl yr yL yR).birthday) (Max.max (Ordinal.l …
      -/
    · exact lt_max_of_lt_left ((nadd_le_nadd_right hj _).trans_lt (lt_lsub _ _))
      /-
        🎉 no goals
      -/
      /-
        case e_a.a.inr.intro
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        i : Ordinal.{u_1}
        hi : LT.lt i (SetTheory.PGame.mk xl xr xL xR).birthday
        j : (SetTheory.PGame.mk xl xr xL xR).RightMoves
        hj : LE.le i ((SetTheory.PGame.mk xl xr xL xR).moveRight j).birthday
        ⊢ LT.lt (i.nadd (SetTheory.PGame.mk yl yr yL yR).birthday) (Max.max (Ordinal.l …
      -/
    · exact lt_max_of_lt_right ((nadd_le_nadd_right hj _).trans_lt (lt_lsub _ _))
      /-
        🎉 no goals
      -/
      /-
        case e_a.a.inl.intro
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        i : Ordinal.{u_1}
        hi : LT.lt i (SetTheory.PGame.mk yl yr yL yR).birthday
        j : (SetTheory.PGame.mk yl yr yL yR).LeftMoves
        hj : LE.le i ((SetTheory.PGame.mk yl yr yL yR).moveLeft j).birthday
        ⊢ LT.lt ((SetTheory.PGame.mk xl xr xL xR).birthday.nadd i) (Max.max (Ordinal.l …
      -/
    · exact lt_max_of_lt_left ((nadd_le_nadd_left hj _).trans_lt (lt_lsub _ _))
      /-
        🎉 no goals
      -/
      /-
        case e_a.a.inr.intro
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        i : Ordinal.{u_1}
        hi : LT.lt i (SetTheory.PGame.mk yl yr yL yR).birthday
        j : (SetTheory.PGame.mk yl yr yL yR).RightMoves
        hj : LE.le i ((SetTheory.PGame.mk yl yr yL yR).moveRight j).birthday
        ⊢ LT.lt ((SetTheory.PGame.mk xl xr xL xR).birthday.nadd i) (Max.max (Ordinal.l …
      -/
    · exact lt_max_of_lt_right ((nadd_le_nadd_left hj _).trans_lt (lt_lsub _ _))
      /-
        🎉 no goals
      -/
termination_by a b => (a, b)


@[simp]
theorem birthday_sub (x y : PGame) : (x - y).birthday = x.birthday ♯ y.birthday := by
  /-
    x y : SetTheory.PGame
    ⊢ Eq (HSub.hSub x y).birthday (x.birthday.nadd y.birthday)
  -/
  apply (birthday_add x _).trans
  /-
    x y : SetTheory.PGame
    ⊢ Eq (x.birthday.nadd (Neg.neg y).birthday) (x.birthday.nadd y.birthday)
  -/
  rw [birthday_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem birthday_natCast : ∀ n : ℕ, birthday n = n
  | 0 => birthday_zero
                /-
                  n : Nat
                  ⊢ Eq (↑(HAdd.hAdd n 1)).birthday ↑(HAdd.hAdd n 1)
                -/
  | n + 1 => by simp [birthday_natCast]
                /-
                  🎉 no goals
                -/


/-- The birthday of a game is defined as the least birthday among all pre-games that define it. -/
noncomputable def birthday (x : Game.{u}) : Ordinal.{u} :=
  sInf (PGame.birthday '' (Quotient.mk' ⁻¹' {x}))


theorem birthday_eq_pGameBirthday (x : Game) :
    ∃ y : PGame.{u}, ⟦y⟧ = x ∧ y.birthday = birthday x := by
  /-
    x : SetTheory.Game
    ⊢ Exists fun y => And (Eq (Quotient.mk SetTheory.PGame.setoid y) x) (Eq y.birt …
  -/
  refine csInf_mem (Set.image_nonempty.2 ?_)
  /-
    x : SetTheory.Game
    ⊢ Set.Nonempty fun y => Eq (Quotient.mk SetTheory.PGame.setoid y) x
  -/
  exact ⟨_, x.out_eq⟩
  /-
    🎉 no goals
  -/


theorem birthday_quot_le_pGameBirthday  (x : PGame) : birthday ⟦x⟧ ≤ x.birthday :=
  csInf_le' ⟨x, rfl, rfl⟩


@[simp]
theorem birthday_zero : birthday 0 = 0 := by
  /-
    ⊢ Eq (SetTheory.Game.birthday 0) 0
  -/
  rw [← Ordinal.le_zero, ← PGame.birthday_zero]
  /-
    ⊢ LE.le (SetTheory.Game.birthday 0) (SetTheory.PGame.birthday 0)
  -/
  exact birthday_quot_le_pGameBirthday  _
  /-
    🎉 no goals
  -/


@[simp]
theorem birthday_eq_zero {x : Game} : birthday x = 0 ↔ x = 0 := by
  /-
    x : SetTheory.Game
    ⊢ Iff (Eq x.birthday 0) (Eq x 0)
  -/
  constructor
    /-
      case mp
      x : SetTheory.Game
      ⊢ Eq x.birthday 0 → Eq x 0
    -/
  · intro h
    /-
      case mp
      x : SetTheory.Game
      h : Eq x.birthday 0
      ⊢ Eq x 0
    -/
    let ⟨y, hy₁, hy₂⟩ := birthday_eq_pGameBirthday x
    /-
      case mp
      x : SetTheory.Game
      h : Eq x.birthday 0
      y : SetTheory.PGame
      hy₁ : Eq (Quotient.mk SetTheory.PGame.setoid y) x
      hy₂ : Eq y.birthday x.birthday
      ⊢ Eq x 0
    -/
    rw [← hy₁]
    /-
      case mp
      x : SetTheory.Game
      h : Eq x.birthday 0
      y : SetTheory.PGame
      hy₁ : Eq (Quotient.mk SetTheory.PGame.setoid y) x
      hy₂ : Eq y.birthday x.birthday
      ⊢ Eq (Quotient.mk SetTheory.PGame.setoid y) 0
    -/
    rw [h, PGame.birthday_eq_zero] at hy₂
    /-
      case mp
      x : SetTheory.Game
      h : Eq x.birthday 0
      y : SetTheory.PGame
      hy₁ : Eq (Quotient.mk SetTheory.PGame.setoid y) x
      hy₂ : And (IsEmpty y.LeftMoves) (IsEmpty y.RightMoves)
      ⊢ Eq (Quotient.mk SetTheory.PGame.setoid y) 0
    -/
    exact PGame.game_eq (@PGame.Equiv.isEmpty _ hy₂.1 hy₂.2)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      x : SetTheory.Game
      ⊢ Eq x 0 → Eq x.birthday 0
    -/
  · rintro rfl
    /-
      case mpr
      ⊢ Eq (SetTheory.Game.birthday 0) 0
    -/
    exact birthday_zero
    /-
      🎉 no goals
    -/


@[simp]
theorem birthday_ordinalToGame (o : Ordinal) : birthday o.toGame = o := by
  /-
    o : Ordinal.{u_1}
    ⊢ Eq (Ordinal.toGame o).birthday o
  -/
  apply le_antisymm
    /-
      case a
      o : Ordinal.{u_1}
      ⊢ LE.le (Ordinal.toGame o).birthday o
    -/
  · conv_rhs => rw [← PGame.birthday_ordinalToPGame o]
    /-
      case a
      o : Ordinal.{u_1}
      ⊢ LE.le (Ordinal.toGame o).birthday o.toPGame.birthday
    -/
    apply birthday_quot_le_pGameBirthday
    /-
      🎉 no goals
    -/
    /-
      case a
      o : Ordinal.{u_1}
      ⊢ LE.le o (Ordinal.toGame o).birthday
    -/
  · let ⟨x, hx₁, hx₂⟩ := birthday_eq_pGameBirthday o.toGame
    /-
      case a
      o : Ordinal.{u_1}
      x : SetTheory.PGame
      hx₁ : Eq (Quotient.mk SetTheory.PGame.setoid x) (Ordinal.toGame o)
      hx₂ : Eq x.birthday (Ordinal.toGame o).birthday
      ⊢ LE.le o (Ordinal.toGame o).birthday
    -/
    rw [← hx₂, ← toPGame_le_iff]
    /-
      case a
      o : Ordinal.{u_1}
      x : SetTheory.PGame
      hx₁ : Eq (Quotient.mk SetTheory.PGame.setoid x) (Ordinal.toGame o)
      hx₂ : Eq x.birthday (Ordinal.toGame o).birthday
      ⊢ LE.le o.toPGame x.birthday.toPGame
    -/
    rw [← mk_toPGame, ← PGame.equiv_iff_game_eq] at hx₁
    /-
      case a
      o : Ordinal.{u_1}
      x : SetTheory.PGame
      hx₁ : HasEquiv.Equiv x o.toPGame
      hx₂ : Eq x.birthday (Ordinal.toGame o).birthday
      ⊢ LE.le o.toPGame x.birthday.toPGame
    -/
    exact hx₁.2.trans (PGame.le_birthday x)
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem birthday_natCast (n : ℕ) : birthday n = n := by
  /-
    n : Nat
    ⊢ Eq (↑n).birthday ↑n
  -/
  rw [← toGame_natCast]
  /-
    n : Nat
    ⊢ Eq (Ordinal.toGame ↑n).birthday ↑n
  -/
  exact birthday_ordinalToGame _
  /-
    🎉 no goals
  -/

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem birthday_ofNat (n : ℕ) [n.AtLeastTwo] :
    birthday (no_index (OfNat.ofNat n)) = OfNat.ofNat n :=
  birthday_natCast n


@[simp]
theorem birthday_one : birthday 1 = 1 := by
  /-
    ⊢ Eq (SetTheory.Game.birthday 1) 1
  -/
  rw [← Nat.cast_one, birthday_natCast, Nat.cast_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem birthday_star : birthday ⟦PGame.star⟧ = 1 := by
  /-
    ⊢ Eq (SetTheory.Game.birthday (Quotient.mk SetTheory.PGame.setoid SetTheory.PG …
  -/
  apply le_antisymm
    /-
      case a
      ⊢ LE.le (SetTheory.Game.birthday (Quotient.mk SetTheory.PGame.setoid SetTheory …
    -/
  · rw [← PGame.birthday_star]
    /-
      case a
      ⊢ LE.le (SetTheory.Game.birthday (Quotient.mk SetTheory.PGame.setoid SetTheory …
    -/
    exact birthday_quot_le_pGameBirthday  _
    /-
      🎉 no goals
    -/
  · rw [Ordinal.one_le_iff_ne_zero, ne_eq, birthday_eq_zero, Game.zero_def,
      ← PGame.equiv_iff_game_eq]
    /-
      case a
      ⊢ Not (HasEquiv.Equiv SetTheory.PGame.star 0)
    -/
    exact PGame.star_fuzzy_zero.not_equiv
    /-
      🎉 no goals
    -/


private theorem birthday_neg' (x : Game) : (-x).birthday ≤ x.birthday := by
  /-
    x : SetTheory.Game
    ⊢ LE.le (Neg.neg x).birthday x.birthday
  -/
  let ⟨y, hy₁, hy₂⟩ := birthday_eq_pGameBirthday x
  /-
    x : SetTheory.Game
    y : SetTheory.PGame
    hy₁ : Eq (Quotient.mk SetTheory.PGame.setoid y) x
    hy₂ : Eq y.birthday x.birthday
    ⊢ LE.le (Neg.neg x).birthday x.birthday
  -/
  rw [← hy₂, ← PGame.birthday_neg y]
  /-
    x : SetTheory.Game
    y : SetTheory.PGame
    hy₁ : Eq (Quotient.mk SetTheory.PGame.setoid y) x
    hy₂ : Eq y.birthday x.birthday
    ⊢ LE.le (Neg.neg x).birthday (Neg.neg y).birthday
  -/
  conv_lhs => rw [← hy₁]
  /-
    x : SetTheory.Game
    y : SetTheory.PGame
    hy₁ : Eq (Quotient.mk SetTheory.PGame.setoid y) x
    hy₂ : Eq y.birthday x.birthday
    ⊢ LE.le (Neg.neg (Quotient.mk SetTheory.PGame.setoid y)).birthday (Neg.neg y). …
  -/
  apply birthday_quot_le_pGameBirthday
  /-
    🎉 no goals
  -/


@[simp]
theorem birthday_neg (x : Game) : (-x).birthday = x.birthday := by
  /-
    x : SetTheory.Game
    ⊢ Eq (Neg.neg x).birthday x.birthday
  -/
  apply le_antisymm (birthday_neg' x)
  /-
    x : SetTheory.Game
    ⊢ LE.le x.birthday (Neg.neg x).birthday
  -/
  conv_lhs => rw [← neg_neg x]
  /-
    x : SetTheory.Game
    ⊢ LE.le (Neg.neg (Neg.neg x)).birthday (Neg.neg x).birthday
  -/
  exact birthday_neg' _
  /-
    🎉 no goals
  -/


theorem le_birthday (x : Game) : x ≤ x.birthday.toGame := by
  /-
    x : SetTheory.Game
    ⊢ LE.le x (Ordinal.toGame x.birthday)
  -/
  let ⟨y, hy₁, hy₂⟩ := birthday_eq_pGameBirthday x
  /-
    x : SetTheory.Game
    y : SetTheory.PGame
    hy₁ : Eq (Quotient.mk SetTheory.PGame.setoid y) x
    hy₂ : Eq y.birthday x.birthday
    ⊢ LE.le x (Ordinal.toGame x.birthday)
  -/
  rw [← hy₁]
  /-
    x : SetTheory.Game
    y : SetTheory.PGame
    hy₁ : Eq (Quotient.mk SetTheory.PGame.setoid y) x
    hy₂ : Eq y.birthday x.birthday
    ⊢ LE.le (Quotient.mk SetTheory.PGame.setoid y) (Ordinal.toGame (SetTheory.Game …
  -/
  apply (y.le_birthday).trans
  /-
    x : SetTheory.Game
    y : SetTheory.PGame
    hy₁ : Eq (Quotient.mk SetTheory.PGame.setoid y) x
    hy₂ : Eq y.birthday x.birthday
    ⊢ LE.le y.birthday.toPGame (SetTheory.Game.birthday (Quotient.mk SetTheory.PGa …
  -/
  rw [toPGame_le_iff, hy₁, hy₂]
  /-
    🎉 no goals
  -/


theorem neg_birthday_le (x : Game) : -x.birthday.toGame ≤ x := by
  /-
    x : SetTheory.Game
    ⊢ LE.le (Neg.neg (Ordinal.toGame x.birthday)) x
  -/
  rw [neg_le, ← birthday_neg]
  /-
    x : SetTheory.Game
    ⊢ LE.le (Neg.neg x) (Ordinal.toGame (Neg.neg x).birthday)
  -/
  exact le_birthday _
  /-
    🎉 no goals
  -/


theorem birthday_add_le (x y : Game) : (x + y).birthday ≤ x.birthday ♯ y.birthday := by
  /-
    x y : SetTheory.Game
    ⊢ LE.le (HAdd.hAdd x y).birthday (x.birthday.nadd y.birthday)
  -/
  let ⟨a, ha₁, ha₂⟩ := birthday_eq_pGameBirthday x
  /-
    x y : SetTheory.Game
    a : SetTheory.PGame
    ha₁ : Eq (Quotient.mk SetTheory.PGame.setoid a) x
    ha₂ : Eq a.birthday x.birthday
    ⊢ LE.le (HAdd.hAdd x y).birthday (x.birthday.nadd y.birthday)
  -/
  let ⟨b, hb₁, hb₂⟩ := birthday_eq_pGameBirthday y
  /-
    x y : SetTheory.Game
    a : SetTheory.PGame
    ha₁ : Eq (Quotient.mk SetTheory.PGame.setoid a) x
    ha₂ : Eq a.birthday x.birthday
    b : SetTheory.PGame
    hb₁ : Eq (Quotient.mk SetTheory.PGame.setoid b) y
    hb₂ : Eq b.birthday y.birthday
    ⊢ LE.le (HAdd.hAdd x y).birthday (x.birthday.nadd y.birthday)
  -/
  rw [← ha₂, ← hb₂, ← ha₁, ← hb₁, ← PGame.birthday_add]
  /-
    x y : SetTheory.Game
    a : SetTheory.PGame
    ha₁ : Eq (Quotient.mk SetTheory.PGame.setoid a) x
    ha₂ : Eq a.birthday x.birthday
    b : SetTheory.PGame
    hb₁ : Eq (Quotient.mk SetTheory.PGame.setoid b) y
    hb₂ : Eq b.birthday y.birthday
    ⊢ LE.le (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid a) (Quotient.mk SetTheo …
  -/
  exact birthday_quot_le_pGameBirthday  _
  /-
    🎉 no goals
  -/


theorem birthday_sub_le (x y : Game) : (x - y).birthday ≤ x.birthday ♯ y.birthday := by
  /-
    x y : SetTheory.Game
    ⊢ LE.le (HSub.hSub x y).birthday (x.birthday.nadd y.birthday)
  -/
  apply (birthday_add_le x _).trans_eq
  /-
    x y : SetTheory.Game
    ⊢ Eq (x.birthday.nadd (Neg.neg y).birthday) (x.birthday.nadd y.birthday)
  -/
  rw [birthday_neg]
  /-
    🎉 no goals
  -/

/- The bound `(x * y).birthday ≤ x.birthday ⨳ y.birthday` is currently an open problem. See
  https://mathoverflow.net/a/476829/147705. -/


/-- Games with bounded birthday are a small set. -/
theorem small_setOf_birthday_lt (o : Ordinal) : Small.{u} {x : Game.{u} // birthday x < o} := by
  induction o using Ordinal.induction with | h o IH =>
  let S := ⋃ a ∈ Set.Iio o, {x : Game.{u} | birthday x < a}
  let H : Small.{u} S := @small_biUnion _ _ _ _ _ IH
  obtain rfl | ⟨a, rfl⟩ | ho := zero_or_succ_or_limit o
  · simp_rw [Ordinal.not_lt_zero]
    exact small_empty
  · simp_rw [Order.lt_succ_iff, le_iff_lt_or_eq]
    convert small_union.{u} {x | birthday x < a} {x | birthday x = a}
    · exact IH _ (Order.lt_succ a)
    · let f (g : Set S × Set S) : Game := ⟦PGame.mk _ _
        (fun x ↦ ((equivShrink g.1).symm x).1.1.out) (fun x ↦ ((equivShrink g.2).symm x).1.1.out)⟧
      suffices {x | x.birthday = a} ⊆ Set.range f from small_subset this
      rintro x rfl
      obtain ⟨y, rfl, hy'⟩ := birthday_eq_pGameBirthday x
      refine ⟨⟨{z | ∃ i, ⟦y.moveLeft i⟧ = z.1}, {z | ∃ i, ⟦y.moveRight i⟧ = z.1}⟩, ?_⟩
      apply PGame.game_eq <| PGame.Equiv.of_exists _ _ _ _ <;> intro i
      · obtain ⟨j, hj⟩ := ((equivShrink _).symm i).2
        exact ⟨j, by simp [PGame.equiv_iff_game_eq, hj]⟩
      · obtain ⟨j, hj⟩ := ((equivShrink _).symm i).2
        exact ⟨j, by simp [PGame.equiv_iff_game_eq, hj]⟩
      · refine ⟨equivShrink _ ⟨⟨⟦y.moveLeft i⟧, ?_⟩, i, rfl⟩, by simpa using Quotient.mk_out _⟩
        suffices ∃ b ≤ y.birthday, birthday ⟦y.moveLeft i⟧ < b by simpa [S, hy'] using this
        refine ⟨_, le_rfl, ?_⟩
        exact (birthday_quot_le_pGameBirthday _).trans_lt (PGame.birthday_moveLeft_lt i)
      · refine ⟨equivShrink _ ⟨⟨⟦y.moveRight i⟧, ?_⟩, i, rfl⟩, by simpa using Quotient.mk_out _⟩
        suffices ∃ b ≤ y.birthday, birthday ⟦y.moveRight i⟧ < b by simpa [S, hy'] using this
        refine ⟨_, le_rfl, ?_⟩
        exact (birthday_quot_le_pGameBirthday _).trans_lt (PGame.birthday_moveRight_lt i)
  · convert H
    change birthday _ < o ↔ ∃ a, _
    simpa using lt_limit ho



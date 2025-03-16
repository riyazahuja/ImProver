/-- Converts an ordinal into the corresponding pre-game. -/
noncomputable def toPGame (o : Ordinal.{u}) : PGame.{u} :=
  ⟨o.toType, PEmpty, fun x => ((enumIsoToType o).symm x).val.toPGame, PEmpty.elim⟩
termination_by o
/-
  o : Ordinal.{u}
  x : o.toType
  ⊢ LT.lt (↑(o.enumIsoToType.symm x)) o
-/
decreasing_by exact ((enumIsoToType o).symm x).prop
/-
  🎉 no goals
-/


@[deprecated "No deprecation message was provided." (since := "2024-09-22")]
theorem toPGame_def (o : Ordinal) : o.toPGame =
    ⟨o.toType, PEmpty, fun x => ((enumIsoToType o).symm x).val.toPGame, PEmpty.elim⟩ := by
  /-
    o : Ordinal.{u_1}
    ⊢ Eq o.toPGame (SetTheory.PGame.mk o.toType PEmpty.{u_1 + 1} (fun x => (↑(o.en …
  -/
  rw [toPGame]
  /-
    🎉 no goals
  -/


@[simp]
theorem toPGame_leftMoves (o : Ordinal) : o.toPGame.LeftMoves = o.toType := by
  /-
    o : Ordinal.{u_1}
    ⊢ Eq o.toPGame.LeftMoves o.toType
  -/
  rw [toPGame, LeftMoves]
  /-
    🎉 no goals
  -/


@[simp]
theorem toPGame_rightMoves (o : Ordinal) : o.toPGame.RightMoves = PEmpty := by
  /-
    o : Ordinal.{u_1}
    ⊢ Eq o.toPGame.RightMoves PEmpty.{u_1 + 1}
  -/
  rw [toPGame, RightMoves]
  /-
    🎉 no goals
  -/


instance isEmpty_zero_toPGame_leftMoves : IsEmpty (toPGame 0).LeftMoves := by
  /-
    ⊢ IsEmpty (Ordinal.toPGame 0).LeftMoves
  -/
  rw [toPGame_leftMoves]; infer_instance
                          /-
                            🎉 no goals
                          -/


instance isEmpty_toPGame_rightMoves (o : Ordinal) : IsEmpty o.toPGame.RightMoves := by
  /-
    o : Ordinal.{u_1}
    ⊢ IsEmpty o.toPGame.RightMoves
  -/
  rw [toPGame_rightMoves]; infer_instance
                           /-
                             🎉 no goals
                           -/


/-- Converts an ordinal less than `o` into a move for the `PGame` corresponding to `o`, and vice
versa. -/
noncomputable def toLeftMovesToPGame {o : Ordinal} : Set.Iio o ≃ o.toPGame.LeftMoves :=
  (enumIsoToType o).toEquiv.trans (Equiv.cast (toPGame_leftMoves o).symm)


@[simp]
theorem toLeftMovesToPGame_symm_lt {o : Ordinal} (i : o.toPGame.LeftMoves) :
    ↑(toLeftMovesToPGame.symm i) < o :=
  (toLeftMovesToPGame.symm i).prop


@[nolint unusedHavesSuffices]
theorem toPGame_moveLeft_hEq {o : Ordinal} :
    HEq o.toPGame.moveLeft fun x : o.toType => ((enumIsoToType o).symm x).val.toPGame := by
  /-
    o : Ordinal.{u_1}
    ⊢ HEq o.toPGame.moveLeft fun x => (↑(o.enumIsoToType.symm x)).toPGame
  -/
  rw [toPGame]
  /-
    o : Ordinal.{u_1}
    ⊢ HEq (SetTheory.PGame.mk o.toType PEmpty.{u_1 + 1} (fun x => (↑(o.enumIsoToTy …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem toPGame_moveLeft' {o : Ordinal} (i) :
    o.toPGame.moveLeft i = (toLeftMovesToPGame.symm i).val.toPGame :=
  (congr_heq toPGame_moveLeft_hEq.symm (cast_heq _ i)).symm


theorem toPGame_moveLeft {o : Ordinal} (i) :
                                                                    /-
                                                                      o : Ordinal.{u_1}
                                                                      i : ↑(Set.Iio o)
                                                                      ⊢ Eq (o.toPGame.moveLeft (Ordinal.toLeftMovesToPGame i)) (↑i).toPGame
                                                                    -/
    o.toPGame.moveLeft (toLeftMovesToPGame i) = i.val.toPGame := by simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- `0.toPGame` has the same moves as `0`. -/
noncomputable def zeroToPGameRelabelling : toPGame 0 ≡r 0 :=
  Relabelling.isEmpty _


theorem toPGame_zero : toPGame 0 ≈ 0 :=
  zeroToPGameRelabelling.equiv


noncomputable instance uniqueOneToPGameLeftMoves : Unique (toPGame 1).LeftMoves :=
  (Equiv.cast <| toPGame_leftMoves 1).unique


@[simp]
theorem one_toPGame_leftMoves_default_eq :
    (default : (toPGame 1).LeftMoves) = @toLeftMovesToPGame 1 ⟨0, Set.mem_Iio.mpr zero_lt_one⟩ :=
  rfl


@[simp]
theorem to_leftMoves_one_toPGame_symm (i) :
    (@toLeftMovesToPGame 1).symm i = ⟨0, Set.mem_Iio.mpr zero_lt_one⟩ := by
  /-
    i : (Ordinal.toPGame 1).LeftMoves
    ⊢ Eq (Ordinal.toLeftMovesToPGame.symm i) ⟨0, ⋯⟩
  -/
  simp [eq_iff_true_of_subsingleton]
  /-
    🎉 no goals
  -/


                                                                            /-
                                                                              x : (Ordinal.toPGame 1).LeftMoves
                                                                              ⊢ Eq ((Ordinal.toPGame 1).moveLeft x) (Ordinal.toPGame 0)
                                                                            -/
theorem one_toPGame_moveLeft (x) : (toPGame 1).moveLeft x = toPGame 0 := by simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/-- `1.toPGame` has the same moves as `1`. -/
noncomputable def oneToPGameRelabelling : toPGame 1 ≡r 1 :=
  ⟨Equiv.ofUnique _ _, Equiv.equivOfIsEmpty _ _, fun i => by
    /-
      i : (Ordinal.toPGame 1).LeftMoves
      ⊢ ((Ordinal.toPGame 1).moveLeft i).Relabelling (SetTheory.PGame.moveLeft 1 ((E …
    -/
    simpa using zeroToPGameRelabelling, isEmptyElim⟩
    /-
      🎉 no goals
    -/


theorem toPGame_one : toPGame 1 ≈ 1 :=
  oneToPGameRelabelling.equiv


theorem toPGame_lf {a b : Ordinal} (h : a < b) : a.toPGame ⧏ b.toPGame := by
  /-
    a b : Ordinal.{u_1}
    h : LT.lt a b
    ⊢ a.toPGame.LF b.toPGame
  -/
  convert moveLeft_lf (toLeftMovesToPGame ⟨a, h⟩); rw [toPGame_moveLeft]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem toPGame_le {a b : Ordinal} (h : a ≤ b) : a.toPGame ≤ b.toPGame := by
  /-
    a b : Ordinal.{u_1}
    h : LE.le a b
    ⊢ LE.le a.toPGame b.toPGame
  -/
  refine le_iff_forall_lf.2 ⟨fun i => ?_, isEmptyElim⟩
  /-
    a b : Ordinal.{u_1}
    h : LE.le a b
    i : a.toPGame.LeftMoves
    ⊢ (a.toPGame.moveLeft i).LF b.toPGame
  -/
  rw [toPGame_moveLeft']
  /-
    a b : Ordinal.{u_1}
    h : LE.le a b
    i : a.toPGame.LeftMoves
    ⊢ (↑(Ordinal.toLeftMovesToPGame.symm i)).toPGame.LF b.toPGame
  -/
  exact toPGame_lf ((toLeftMovesToPGame_symm_lt i).trans_le h)
  /-
    🎉 no goals
  -/


theorem toPGame_lt {a b : Ordinal} (h : a < b) : a.toPGame < b.toPGame :=
  ⟨toPGame_le h.le, toPGame_lf h⟩


theorem toPGame_nonneg (a : Ordinal) : 0 ≤ a.toPGame :=
  zeroToPGameRelabelling.ge.trans <| toPGame_le <| Ordinal.zero_le a


@[simp]
theorem toPGame_lf_iff {a b : Ordinal} : a.toPGame ⧏ b.toPGame ↔ a < b :=
      /-
        a b : Ordinal.{u_1}
        ⊢ a.toPGame.LF b.toPGame → LT.lt a b
      -/
  ⟨by contrapose; rw [not_lt, not_lf]; exact toPGame_le, toPGame_lf⟩
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem toPGame_le_iff {a b : Ordinal} : a.toPGame ≤ b.toPGame ↔ a ≤ b :=
      /-
        a b : Ordinal.{u_1}
        ⊢ LE.le a.toPGame b.toPGame → LE.le a b
      -/
  ⟨by contrapose; rw [not_le, PGame.not_le]; exact toPGame_lf, toPGame_le⟩
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem toPGame_lt_iff {a b : Ordinal} : a.toPGame < b.toPGame ↔ a < b :=
      /-
        a b : Ordinal.{u_1}
        ⊢ LT.lt a.toPGame b.toPGame → LT.lt a b
      -/
  ⟨by contrapose; rw [not_lt]; exact fun h => not_lt_of_le (toPGame_le h), toPGame_lt⟩
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem toPGame_equiv_iff {a b : Ordinal} : (a.toPGame ≈ b.toPGame) ↔ a = b := by
  -- Porting note: was `rw [PGame.Equiv]`
  /-
    a b : Ordinal.{u_1}
    ⊢ Iff (HasEquiv.Equiv a.toPGame b.toPGame) (Eq a b)
  -/
  change _ ≤_ ∧ _ ≤ _ ↔ _
  /-
    a b : Ordinal.{u_1}
    ⊢ Iff (And (LE.le a.toPGame b.toPGame) (LE.le b.toPGame a.toPGame)) (Eq a b)
  -/
  rw [le_antisymm_iff, toPGame_le_iff, toPGame_le_iff]
  /-
    🎉 no goals
  -/


theorem toPGame_injective : Function.Injective Ordinal.toPGame := fun _ _ h =>
  toPGame_equiv_iff.1 <| equiv_of_eq h


@[simp]
theorem toPGame_inj {a b : Ordinal} : a.toPGame = b.toPGame ↔ a = b :=
  toPGame_injective.eq_iff


@[deprecated (since := "2024-12-29")] alias toPGame_eq_iff := toPGame_inj


/-- The order embedding version of `toPGame`. -/
@[simps]
noncomputable def toPGameEmbedding : Ordinal.{u} ↪o PGame.{u} where
  toFun := Ordinal.toPGame
  inj' := toPGame_injective
  map_rel_iff' := @toPGame_le_iff


/-- Converts an ordinal into the corresponding game. -/
noncomputable def toGame : Ordinal.{u} ↪o Game.{u} where
  toFun o := ⟦o.toPGame⟧
                 /-
                   a b : Ordinal.{u}
                   ⊢ Eq ((fun o => Quotient.mk SetTheory.PGame.setoid o.toPGame) a) ((fun o => Qu …
                 -/
  inj' a b := by simpa [AntisymmRel] using le_antisymm
                 /-
                   🎉 no goals
                 -/
  map_rel_iff' := toPGame_le_iff


@[simp]
theorem mk_toPGame (o : Ordinal) : ⟦o.toPGame⟧ = o.toGame :=
  rfl


@[deprecated toGame (since := "2024-11-23")]
alias toGameEmbedding := toGame


@[simp]
theorem toGame_zero : toGame 0 = 0 :=
  game_eq toPGame_zero


@[simp]
theorem toGame_one : toGame 1 = 1 :=
  game_eq toPGame_one


theorem toGame_injective : Function.Injective toGame :=
  toGame.injective


@[simp]
theorem toGame_lf_iff {a b : Ordinal} : Game.LF a.toGame b.toGame ↔ a < b :=
  toPGame_lf_iff


theorem toGame_le_iff {a b : Ordinal} : a.toGame ≤ b.toGame ↔ a ≤ b :=
  toPGame_le_iff


theorem toGame_lt_iff {a b : Ordinal} : a.toGame < b.toGame ↔ a < b :=
  toPGame_lt_iff


theorem toGame_inj {a b : Ordinal} : a.toGame = b.toGame ↔ a = b :=
  toGame.inj


@[deprecated (since := "2024-12-29")] alias toGame_eq_iff := toGame_inj


/-- The natural addition of ordinals corresponds to their sum as games. -/
theorem toPGame_nadd (a b : Ordinal) : (a ♯ b).toPGame ≈ a.toPGame + b.toPGame := by
  /-
    a b : Ordinal.{u_1}
    ⊢ HasEquiv.Equiv (a.nadd b).toPGame (HAdd.hAdd a.toPGame b.toPGame)
  -/
  refine ⟨le_of_forall_lf (fun i => ?_) isEmptyElim, le_of_forall_lf (fun i => ?_) isEmptyElim⟩
    /-
      case refine_1
      a b : Ordinal.{u_1}
      i : (a.nadd b).toPGame.LeftMoves
      ⊢ ((a.nadd b).toPGame.moveLeft i).LF (HAdd.hAdd a.toPGame b.toPGame)
    -/
  · rw [toPGame_moveLeft']
    /-
      case refine_1
      a b : Ordinal.{u_1}
      i : (a.nadd b).toPGame.LeftMoves
      ⊢ (↑(Ordinal.toLeftMovesToPGame.symm i)).toPGame.LF (HAdd.hAdd a.toPGame b.toP …
    -/
    rcases lt_nadd_iff.1 (toLeftMovesToPGame_symm_lt i) with (⟨c, hc, hc'⟩ | ⟨c, hc, hc'⟩) <;>
    /-
      case refine_1.inl.intro.intro
      a b : Ordinal.{u_1}
      i : (a.nadd b).toPGame.LeftMoves
      c : Ordinal.{u_1}
      hc : LT.lt c a
      hc' : LE.le (↑(Ordinal.toLeftMovesToPGame.symm i)) (c.nadd b)
      ⊢ (↑(Ordinal.toLeftMovesToPGame.symm i)).toPGame.LF (HAdd.hAdd a.toPGame b.toP …
    -/
    rw [← toPGame_le_iff, le_congr_right (toPGame_nadd _ _)] at hc' <;>
    /-
      case refine_1.inl.intro.intro
      a b : Ordinal.{u_1}
      i : (a.nadd b).toPGame.LeftMoves
      c : Ordinal.{u_1}
      hc : LT.lt c a
      hc' : LE.le (↑(Ordinal.toLeftMovesToPGame.symm i)).toPGame (HAdd.hAdd c.toPGam …
      ⊢ (↑(Ordinal.toLeftMovesToPGame.symm i)).toPGame.LF (HAdd.hAdd a.toPGame b.toP …
    -/
    apply lf_of_le_of_lf hc'
      /-
        case refine_1.inl.intro.intro
        a b : Ordinal.{u_1}
        i : (a.nadd b).toPGame.LeftMoves
        c : Ordinal.{u_1}
        hc : LT.lt c a
        hc' : LE.le (↑(Ordinal.toLeftMovesToPGame.symm i)).toPGame (HAdd.hAdd c.toPGam …
        ⊢ (HAdd.hAdd c.toPGame b.toPGame).LF (HAdd.hAdd a.toPGame b.toPGame)
      -/
    · apply add_lf_add_right
      /-
        case refine_1.inl.intro.intro.h
        a b : Ordinal.{u_1}
        i : (a.nadd b).toPGame.LeftMoves
        c : Ordinal.{u_1}
        hc : LT.lt c a
        hc' : LE.le (↑(Ordinal.toLeftMovesToPGame.symm i)).toPGame (HAdd.hAdd c.toPGam …
        ⊢ c.toPGame.LF a.toPGame
      -/
      rwa [toPGame_lf_iff]
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr.intro.intro
        a b : Ordinal.{u_1}
        i : (a.nadd b).toPGame.LeftMoves
        c : Ordinal.{u_1}
        hc : LT.lt c b
        hc' : LE.le (↑(Ordinal.toLeftMovesToPGame.symm i)).toPGame (HAdd.hAdd a.toPGam …
        ⊢ (HAdd.hAdd a.toPGame c.toPGame).LF (HAdd.hAdd a.toPGame b.toPGame)
      -/
    · apply add_lf_add_left
      /-
        case refine_1.inr.intro.intro.h
        a b : Ordinal.{u_1}
        i : (a.nadd b).toPGame.LeftMoves
        c : Ordinal.{u_1}
        hc : LT.lt c b
        hc' : LE.le (↑(Ordinal.toLeftMovesToPGame.symm i)).toPGame (HAdd.hAdd a.toPGam …
        ⊢ c.toPGame.LF b.toPGame
      -/
      rwa [toPGame_lf_iff]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      a b : Ordinal.{u_1}
      i : (HAdd.hAdd a.toPGame b.toPGame).LeftMoves
      ⊢ ((HAdd.hAdd a.toPGame b.toPGame).moveLeft i).LF (a.nadd b).toPGame
    -/
  · apply leftMoves_add_cases i <;>
    /-
      case refine_2.hl
      a b : Ordinal.{u_1}
      i : (HAdd.hAdd a.toPGame b.toPGame).LeftMoves
      ⊢ ∀ (i : a.toPGame.LeftMoves), ((HAdd.hAdd a.toPGame b.toPGame).moveLeft (SetT …
    -/
    intro i <;>
    /-
      case refine_2.hl
      a b : Ordinal.{u_1}
      i✝ : (HAdd.hAdd a.toPGame b.toPGame).LeftMoves
      i : a.toPGame.LeftMoves
      ⊢ ((HAdd.hAdd a.toPGame b.toPGame).moveLeft (SetTheory.PGame.toLeftMovesAdd (S …
    -/
    let wf := toLeftMovesToPGame_symm_lt i <;>
     /-
       case refine_2.hl
       a b : Ordinal.{u_1}
       i✝ : (HAdd.hAdd a.toPGame b.toPGame).LeftMoves
       i : a.toPGame.LeftMoves
       wf : LT.lt (↑(Ordinal.toLeftMovesToPGame.symm i)) a := Ordinal.toLeftMovesToPG …
       ⊢ ((HAdd.hAdd a.toPGame b.toPGame).moveLeft (SetTheory.PGame.toLeftMovesAdd (S …
     -/
    (try rw [add_moveLeft_inl]) <;>
     /-
       case refine_2.hl
       a b : Ordinal.{u_1}
       i✝ : (HAdd.hAdd a.toPGame b.toPGame).LeftMoves
       i : a.toPGame.LeftMoves
       wf : LT.lt (↑(Ordinal.toLeftMovesToPGame.symm i)) a := Ordinal.toLeftMovesToPG …
       ⊢ (HAdd.hAdd (a.toPGame.moveLeft i) b.toPGame).LF (a.nadd b).toPGame
     -/
    (try rw [add_moveLeft_inr]) <;>
    /-
      case refine_2.hl
      a b : Ordinal.{u_1}
      i✝ : (HAdd.hAdd a.toPGame b.toPGame).LeftMoves
      i : a.toPGame.LeftMoves
      wf : LT.lt (↑(Ordinal.toLeftMovesToPGame.symm i)) a := Ordinal.toLeftMovesToPG …
      ⊢ (HAdd.hAdd (a.toPGame.moveLeft i) b.toPGame).LF (a.nadd b).toPGame
    -/
    rw [toPGame_moveLeft', ← lf_congr_left (toPGame_nadd _ _), toPGame_lf_iff]
      /-
        case refine_2.hl
        a b : Ordinal.{u_1}
        i✝ : (HAdd.hAdd a.toPGame b.toPGame).LeftMoves
        i : a.toPGame.LeftMoves
        wf : LT.lt (↑(Ordinal.toLeftMovesToPGame.symm i)) a := Ordinal.toLeftMovesToPG …
        ⊢ LT.lt ((↑(Ordinal.toLeftMovesToPGame.symm i)).nadd b) (a.nadd b)
      -/
    · exact nadd_lt_nadd_right wf _
      /-
        🎉 no goals
      -/
      /-
        case refine_2.hr
        a b : Ordinal.{u_1}
        i✝ : (HAdd.hAdd a.toPGame b.toPGame).LeftMoves
        i : b.toPGame.LeftMoves
        wf : LT.lt (↑(Ordinal.toLeftMovesToPGame.symm i)) b := Ordinal.toLeftMovesToPG …
        ⊢ LT.lt (a.nadd ↑(Ordinal.toLeftMovesToPGame.symm i)) (a.nadd b)
      -/
    · exact nadd_lt_nadd_left wf _
      /-
        🎉 no goals
      -/
termination_by (a, b)


theorem toGame_nadd (a b : Ordinal) : (a ♯ b).toGame = a.toGame + b.toGame :=
  game_eq (toPGame_nadd a b)


/-- The natural multiplication of ordinals corresponds to their product as pre-games. -/
theorem toPGame_nmul (a b : Ordinal) : (a ⨳ b).toPGame ≈ a.toPGame * b.toPGame := by
  /-
    a b : Ordinal.{u_1}
    ⊢ HasEquiv.Equiv (a.nmul b).toPGame (HMul.hMul a.toPGame b.toPGame)
  -/
  refine ⟨le_of_forall_lf (fun i => ?_) isEmptyElim, le_of_forall_lf (fun i => ?_) isEmptyElim⟩
    /-
      case refine_1
      a b : Ordinal.{u_1}
      i : (a.nmul b).toPGame.LeftMoves
      ⊢ ((a.nmul b).toPGame.moveLeft i).LF (HMul.hMul a.toPGame b.toPGame)
    -/
  · rw [toPGame_moveLeft']
    /-
      case refine_1
      a b : Ordinal.{u_1}
      i : (a.nmul b).toPGame.LeftMoves
      ⊢ (↑(Ordinal.toLeftMovesToPGame.symm i)).toPGame.LF (HMul.hMul a.toPGame b.toP …
    -/
    rcases lt_nmul_iff.1 (toLeftMovesToPGame_symm_lt i) with ⟨c, hc, d, hd, h⟩
    rw [← toPGame_le_iff, le_iff_game_le, mk_toPGame, mk_toPGame, toGame_nadd _ _, toGame_nadd _ _,
      ← le_sub_iff_add_le] at h
    refine lf_of_le_of_lf h <| (lf_congr_left ?_).1 <| moveLeft_lf <| toLeftMovesMul <| Sum.inl
      ⟨toLeftMovesToPGame ⟨c, hc⟩, toLeftMovesToPGame ⟨d, hd⟩⟩
    simp only [mul_moveLeft_inl, toPGame_moveLeft', Equiv.symm_apply_apply, equiv_iff_game_eq,
      quot_sub, quot_add]
    /-
      case refine_1.intro.intro.intro.intro
      a b : Ordinal.{u_1}
      i : (a.nmul b).toPGame.LeftMoves
      c : Ordinal.{u_1}
      hc : LT.lt c a
      d : Ordinal.{u_1}
      hd : LT.lt d b
      h : LE.le (Ordinal.toGame ↑(Ordinal.toLeftMovesToPGame.symm i)) (HSub.hSub (HA …
      ⊢ Eq (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (HMul.hMul c.to …
    -/
    repeat rw [← game_eq (toPGame_nmul _ _)]
    /-
      case refine_1.intro.intro.intro.intro
      a b : Ordinal.{u_1}
      i : (a.nmul b).toPGame.LeftMoves
      c : Ordinal.{u_1}
      hc : LT.lt c a
      d : Ordinal.{u_1}
      hd : LT.lt d b
      h : LE.le (Ordinal.toGame ↑(Ordinal.toLeftMovesToPGame.symm i)) (HSub.hSub (HA …
      ⊢ Eq (HSub.hSub (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (c.nmul b).toPG …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a b : Ordinal.{u_1}
      i : (HMul.hMul a.toPGame b.toPGame).LeftMoves
      ⊢ ((HMul.hMul a.toPGame b.toPGame).moveLeft i).LF (a.nmul b).toPGame
    -/
  · apply leftMoves_mul_cases i _ isEmptyElim
    /-
      a b : Ordinal.{u_1}
      i : (HMul.hMul a.toPGame b.toPGame).LeftMoves
      ⊢ ∀ (ix : a.toPGame.LeftMoves) (iy : b.toPGame.LeftMoves), ((HMul.hMul a.toPGa …
    -/
    intro i j
    rw [mul_moveLeft_inl, toPGame_moveLeft', toPGame_moveLeft', lf_iff_game_lf,
      quot_sub, quot_add, ← Game.not_le, le_sub_iff_add_le]
    /-
      a b : Ordinal.{u_1}
      i✝ : (HMul.hMul a.toPGame b.toPGame).LeftMoves
      i : a.toPGame.LeftMoves
      j : b.toPGame.LeftMoves
      ⊢ Not (LE.le (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (a.nmul b).toPGame …
    -/
    repeat rw [← game_eq (toPGame_nmul _ _)]
    /-
      a b : Ordinal.{u_1}
      i✝ : (HMul.hMul a.toPGame b.toPGame).LeftMoves
      i : a.toPGame.LeftMoves
      j : b.toPGame.LeftMoves
      ⊢ Not (LE.le (HAdd.hAdd (Quotient.mk SetTheory.PGame.setoid (a.nmul b).toPGame …
    -/
    simp_rw [mk_toPGame, ← toGame_nadd]
    /-
      a b : Ordinal.{u_1}
      i✝ : (HMul.hMul a.toPGame b.toPGame).LeftMoves
      i : a.toPGame.LeftMoves
      j : b.toPGame.LeftMoves
      ⊢ Not (LE.le (Ordinal.toGame ((a.nmul b).nadd ((↑(Ordinal.toLeftMovesToPGame.s …
    -/
    apply toPGame_lf (nmul_nadd_lt _ _) <;>
    /-
      a b : Ordinal.{u_1}
      i✝ : (HMul.hMul a.toPGame b.toPGame).LeftMoves
      i : a.toPGame.LeftMoves
      j : b.toPGame.LeftMoves
      ⊢ LT.lt (↑(Ordinal.toLeftMovesToPGame.symm i)) a
    -/
    /-
      🎉 no goals
    -/
    exact toLeftMovesToPGame_symm_lt _
    /-
      🎉 no goals
    -/
termination_by (a, b)


theorem toGame_nmul (a b : Ordinal) : (a ⨳ b).toGame = ⟦a.toPGame * b.toPGame⟧ :=
  game_eq (toPGame_nmul a b)


@[simp] -- used to be a norm_cast lemma
theorem toGame_natCast : ∀ n : ℕ, toGame n = n
  | 0 => Quot.sound (zeroToPGameRelabelling).equiv
  | n + 1 => by
    /-
      n : Nat
      ⊢ Eq (Ordinal.toGame ↑(HAdd.hAdd n 1)) ↑(HAdd.hAdd n 1)
    -/
    have : toGame 1 = 1 := Quot.sound oneToPGameRelabelling.equiv
    /-
      n : Nat
      this : Eq (Ordinal.toGame 1) 1
      ⊢ Eq (Ordinal.toGame ↑(HAdd.hAdd n 1)) ↑(HAdd.hAdd n 1)
    -/
    rw [Nat.cast_add, ← nadd_nat, toGame_nadd, toGame_natCast, Nat.cast_one, this]
    /-
      n : Nat
      this : Eq (Ordinal.toGame 1) 1
      ⊢ Eq (HAdd.hAdd (↑n) 1) ↑(HAdd.hAdd n 1)
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem toPGame_natCast (n : ℕ) : toPGame n ≈ n := by
  /-
    n : Nat
    ⊢ HasEquiv.Equiv (↑n).toPGame ↑n
  -/
  rw [PGame.equiv_iff_game_eq, mk_toPGame, toGame_natCast, quot_natCast]
  /-
    🎉 no goals
  -/



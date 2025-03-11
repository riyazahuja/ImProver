/-- The definition of single-heap nim, which can be viewed as a pile of stones where each player can
  take a positive number of stones from it on their turn. -/
noncomputable def nim (o : Ordinal.{u}) : PGame.{u} :=
  ⟨o.toType, o.toType,
    fun x => nim ((enumIsoToType o).symm x).val,
    fun x => nim ((enumIsoToType o).symm x).val⟩
termination_by o
/-
  o : Ordinal.{u}
  x : o.toType
  ⊢ LT.lt (↑(o.enumIsoToType.symm x)) o
-/
decreasing_by all_goals exact ((enumIsoToType o).symm x).prop
/-
  🎉 no goals
-/


theorem nim_def (o : Ordinal) : nim o =
    ⟨o.toType, o.toType,
      fun x => nim ((enumIsoToType o).symm x).val,
      fun x => nim ((enumIsoToType o).symm x).val⟩ := by
  /-
    o : Ordinal.{u_1}
    ⊢ Eq (SetTheory.PGame.nim o) (SetTheory.PGame.mk o.toType o.toType (fun x => S …
  -/
  rw [nim]
  /-
    🎉 no goals
  -/


                                                                         /-
                                                                           o : Ordinal.{u_1}
                                                                           ⊢ Eq (SetTheory.PGame.nim o).LeftMoves o.toType
                                                                         -/
theorem leftMoves_nim (o : Ordinal) : (nim o).LeftMoves = o.toType := by rw [nim_def]; rfl
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/

                                                                           /-
                                                                             o : Ordinal.{u_1}
                                                                             ⊢ Eq (SetTheory.PGame.nim o).RightMoves o.toType
                                                                           -/
theorem rightMoves_nim (o : Ordinal) : (nim o).RightMoves = o.toType := by rw [nim_def]; rfl
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


theorem moveLeft_nim_hEq (o : Ordinal) :
                                                                                  /-
                                                                                    o : Ordinal.{u_1}
                                                                                    ⊢ HEq (SetTheory.PGame.nim o).moveLeft fun i => SetTheory.PGame.nim ↑(o.enumIs …
                                                                                  -/
    HEq (nim o).moveLeft fun i : o.toType => nim ((enumIsoToType o).symm i) := by rw [nim_def]; rfl
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


theorem moveRight_nim_hEq (o : Ordinal) :
                                                                                   /-
                                                                                     o : Ordinal.{u_1}
                                                                                     ⊢ HEq (SetTheory.PGame.nim o).moveRight fun i => SetTheory.PGame.nim ↑(o.enumI …
                                                                                   -/
    HEq (nim o).moveRight fun i : o.toType => nim ((enumIsoToType o).symm i) := by rw [nim_def]; rfl
                                                                                                 /-
                                                                                                   🎉 no goals
                                                                                                 -/


/-- Turns an ordinal less than `o` into a left move for `nim o` and vice versa. -/
noncomputable def toLeftMovesNim {o : Ordinal} : Set.Iio o ≃ (nim o).LeftMoves :=
  (enumIsoToType o).toEquiv.trans (Equiv.cast (leftMoves_nim o).symm)


/-- Turns an ordinal less than `o` into a right move for `nim o` and vice versa. -/
noncomputable def toRightMovesNim {o : Ordinal} : Set.Iio o ≃ (nim o).RightMoves :=
  (enumIsoToType o).toEquiv.trans (Equiv.cast (rightMoves_nim o).symm)


@[simp]
theorem toLeftMovesNim_symm_lt {o : Ordinal} (i : (nim o).LeftMoves) :
    toLeftMovesNim.symm i < o :=
  (toLeftMovesNim.symm i).prop


@[simp]
theorem toRightMovesNim_symm_lt {o : Ordinal} (i : (nim o).RightMoves) :
    toRightMovesNim.symm i < o :=
  (toRightMovesNim.symm i).prop


@[simp]
theorem moveLeft_nim {o : Ordinal} (i) : (nim o).moveLeft i = nim (toLeftMovesNim.symm i).val :=
  (congr_heq (moveLeft_nim_hEq o).symm (cast_heq _ i)).symm


@[deprecated moveLeft_nim (since := "2024-10-30")]
alias moveLeft_nim' := moveLeft_nim


theorem moveLeft_toLeftMovesNim {o : Ordinal} (i) :
    (nim o).moveLeft (toLeftMovesNim i) = nim i := by
  /-
    o : Ordinal.{u_1}
    i : ↑(Set.Iio o)
    ⊢ Eq ((SetTheory.PGame.nim o).moveLeft (SetTheory.PGame.toLeftMovesNim i)) (Se …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem moveRight_nim {o : Ordinal} (i) : (nim o).moveRight i = nim (toRightMovesNim.symm i).val :=
  (congr_heq (moveRight_nim_hEq o).symm (cast_heq _ i)).symm


@[deprecated moveRight_nim (since := "2024-10-30")]
alias moveRight_nim' := moveRight_nim


theorem moveRight_toRightMovesNim {o : Ordinal} (i) :
    (nim o).moveRight (toRightMovesNim i) = nim i := by
  /-
    o : Ordinal.{u_1}
    i : ↑(Set.Iio o)
    ⊢ Eq ((SetTheory.PGame.nim o).moveRight (SetTheory.PGame.toRightMovesNim i)) ( …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A recursion principle for left moves of a nim game. -/
@[elab_as_elim]
def leftMovesNimRecOn {o : Ordinal} {P : (nim o).LeftMoves → Sort*} (i : (nim o).LeftMoves)
    (H : ∀ a (H : a < o), P <| toLeftMovesNim ⟨a, H⟩) : P i := by
  /-
    o : Ordinal.{?u.8496}
    P : (SetTheory.PGame.nim o).LeftMoves → Sort u_1
    i : (SetTheory.PGame.nim o).LeftMoves
    H : (a : Ordinal.{?u.8496}) → (H : LT.lt a o) → P (SetTheory.PGame.toLeftMoves …
    ⊢ P i
  -/
  rw [← toLeftMovesNim.apply_symm_apply i]; apply H
                                            /-
                                              🎉 no goals
                                            -/


/-- A recursion principle for right moves of a nim game. -/
@[elab_as_elim]
def rightMovesNimRecOn {o : Ordinal} {P : (nim o).RightMoves → Sort*} (i : (nim o).RightMoves)
    (H : ∀ a (H : a < o), P <| toRightMovesNim ⟨a, H⟩) : P i := by
  /-
    o : Ordinal.{?u.8779}
    P : (SetTheory.PGame.nim o).RightMoves → Sort u_1
    i : (SetTheory.PGame.nim o).RightMoves
    H : (a : Ordinal.{?u.8779}) → (H : LT.lt a o) → P (SetTheory.PGame.toRightMove …
    ⊢ P i
  -/
  rw [← toRightMovesNim.apply_symm_apply i]; apply H
                                             /-
                                               🎉 no goals
                                             -/


instance isEmpty_nim_zero_leftMoves : IsEmpty (nim 0).LeftMoves := by
  /-
    ⊢ IsEmpty (SetTheory.PGame.nim 0).LeftMoves
  -/
  rw [nim_def]
  /-
    ⊢ IsEmpty (SetTheory.PGame.mk (Ordinal.toType 0) (Ordinal.toType 0) (fun x =>  …
  -/
  exact isEmpty_toType_zero
  /-
    🎉 no goals
  -/


instance isEmpty_nim_zero_rightMoves : IsEmpty (nim 0).RightMoves := by
  /-
    ⊢ IsEmpty (SetTheory.PGame.nim 0).RightMoves
  -/
  rw [nim_def]
  /-
    ⊢ IsEmpty (SetTheory.PGame.mk (Ordinal.toType 0) (Ordinal.toType 0) (fun x =>  …
  -/
  exact isEmpty_toType_zero
  /-
    🎉 no goals
  -/


/-- `nim 0` has exactly the same moves as `0`. -/
def nimZeroRelabelling : nim 0 ≡r 0 :=
  Relabelling.isEmpty _


theorem nim_zero_equiv : nim 0 ≈ 0 :=
  Equiv.isEmpty _


noncomputable instance uniqueNimOneLeftMoves : Unique (nim 1).LeftMoves :=
  (Equiv.cast <| leftMoves_nim 1).unique


noncomputable instance uniqueNimOneRightMoves : Unique (nim 1).RightMoves :=
  (Equiv.cast <| rightMoves_nim 1).unique


@[simp]
theorem default_nim_one_leftMoves_eq :
    (default : (nim 1).LeftMoves) = @toLeftMovesNim 1 ⟨0, Set.mem_Iio.mpr zero_lt_one⟩ :=
  rfl


@[simp]
theorem default_nim_one_rightMoves_eq :
    (default : (nim 1).RightMoves) = @toRightMovesNim 1 ⟨0, Set.mem_Iio.mpr zero_lt_one⟩ :=
  rfl


@[simp]
theorem toLeftMovesNim_one_symm (i) :
    (@toLeftMovesNim 1).symm i = ⟨0, Set.mem_Iio.mpr zero_lt_one⟩ := by
  /-
    i : (SetTheory.PGame.nim 1).LeftMoves
    ⊢ Eq (SetTheory.PGame.toLeftMovesNim.symm i) ⟨0, ⋯⟩
  -/
  simp [eq_iff_true_of_subsingleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem toRightMovesNim_one_symm (i) :
    (@toRightMovesNim 1).symm i = ⟨0, Set.mem_Iio.mpr zero_lt_one⟩ := by
  /-
    i : (SetTheory.PGame.nim 1).RightMoves
    ⊢ Eq (SetTheory.PGame.toRightMovesNim.symm i) ⟨0, ⋯⟩
  -/
  simp [eq_iff_true_of_subsingleton]
  /-
    🎉 no goals
  -/


                                                                /-
                                                                  x : (SetTheory.PGame.nim 1).LeftMoves
                                                                  ⊢ Eq ((SetTheory.PGame.nim 1).moveLeft x) (SetTheory.PGame.nim 0)
                                                                -/
theorem nim_one_moveLeft (x) : (nim 1).moveLeft x = nim 0 := by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


                                                                  /-
                                                                    x : (SetTheory.PGame.nim 1).RightMoves
                                                                    ⊢ Eq ((SetTheory.PGame.nim 1).moveRight x) (SetTheory.PGame.nim 0)
                                                                  -/
theorem nim_one_moveRight (x) : (nim 1).moveRight x = nim 0 := by simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- `nim 1` has exactly the same moves as `star`. -/
def nimOneRelabelling : nim 1 ≡r star := by
  /-
    ⊢ (SetTheory.PGame.nim 1).Relabelling SetTheory.PGame.star
  -/
  rw [nim_def]
  /-
    ⊢ (SetTheory.PGame.mk (Ordinal.toType 1) (Ordinal.toType 1) (fun x => SetTheor …
  -/
  refine ⟨?_, ?_, fun i => ?_, fun j => ?_⟩
  /-
    case refine_1
    ⊢ _root_.Equiv (SetTheory.PGame.mk (Ordinal.toType 1) (Ordinal.toType 1) (fun  …
  -/
  any_goals dsimp; apply Equiv.ofUnique
  /-
    case refine_3
    i : (SetTheory.PGame.mk (Ordinal.toType 1) (Ordinal.toType 1) (fun x => SetThe …
    ⊢ ((SetTheory.PGame.mk (Ordinal.toType 1) (Ordinal.toType 1) (fun x => SetTheo …
  -/
  all_goals simpa [enumIsoToType] using nimZeroRelabelling
  /-
    🎉 no goals
  -/


theorem nim_one_equiv : nim 1 ≈ star :=
  nimOneRelabelling.equiv


@[simp]
theorem nim_birthday (o : Ordinal) : (nim o).birthday = o := by
  /-
    o : Ordinal.{u_1}
    ⊢ Eq (SetTheory.PGame.nim o).birthday o
  -/
  induction' o using Ordinal.induction with o IH
  /-
    case h
    o : Ordinal.{u_1}
    IH : ∀ (k : Ordinal.{u_1}), LT.lt k o → Eq (SetTheory.PGame.nim k).birthday k
    ⊢ Eq (SetTheory.PGame.nim o).birthday o
  -/
  rw [nim_def, birthday_def]
  /-
    case h
    o : Ordinal.{u_1}
    IH : ∀ (k : Ordinal.{u_1}), LT.lt k o → Eq (SetTheory.PGame.nim k).birthday k
    ⊢ Eq (Max.max (Ordinal.lsub fun i => ((SetTheory.PGame.mk o.toType o.toType (f …
  -/
  dsimp
  /-
    case h
    o : Ordinal.{u_1}
    IH : ∀ (k : Ordinal.{u_1}), LT.lt k o → Eq (SetTheory.PGame.nim k).birthday k
    ⊢ Eq (Max.max (Ordinal.lsub fun i => (SetTheory.PGame.nim ↑(o.enumIsoToType.sy …
  -/
  rw [max_eq_right le_rfl]
  /-
    case h
    o : Ordinal.{u_1}
    IH : ∀ (k : Ordinal.{u_1}), LT.lt k o → Eq (SetTheory.PGame.nim k).birthday k
    ⊢ Eq (Ordinal.lsub fun i => (SetTheory.PGame.nim ↑(o.enumIsoToType.symm i)).bi …
  -/
  convert lsub_typein o with i
  /-
    case h.e'_2.h.e'_2.h
    o : Ordinal.{u_1}
    IH : ∀ (k : Ordinal.{u_1}), LT.lt k o → Eq (SetTheory.PGame.nim k).birthday k
    i : o.toType
    ⊢ Eq (SetTheory.PGame.nim ↑(o.enumIsoToType.symm i)).birthday ((Ordinal.typein …
  -/
  exact IH _ (typein_lt_self i)
  /-
    🎉 no goals
  -/


@[simp]
theorem neg_nim (o : Ordinal) : -nim o = nim o := by
  /-
    o : Ordinal.{u_1}
    ⊢ Eq (Neg.neg (SetTheory.PGame.nim o)) (SetTheory.PGame.nim o)
  -/
  induction' o using Ordinal.induction with o IH
  /-
    case h
    o : Ordinal.{u_1}
    IH : ∀ (k : Ordinal.{u_1}), LT.lt k o → Eq (Neg.neg (SetTheory.PGame.nim k)) ( …
    ⊢ Eq (Neg.neg (SetTheory.PGame.nim o)) (SetTheory.PGame.nim o)
  -/
                                              /-
                                                🎉 no goals
                                              -/
  rw [nim_def]; dsimp; congr <;> funext i <;> exact IH _ (Ordinal.typein_lt_self i)
                                              /-
                                                🎉 no goals
                                              -/


instance nim_impartial (o : Ordinal) : Impartial (nim o) := by
  /-
    o : Ordinal.{u_1}
    ⊢ (SetTheory.PGame.nim o).Impartial
  -/
  induction' o using Ordinal.induction with o IH
  /-
    case h
    o : Ordinal.{u_1}
    IH : ∀ (k : Ordinal.{u_1}), LT.lt k o → (SetTheory.PGame.nim k).Impartial
    ⊢ (SetTheory.PGame.nim o).Impartial
  -/
  rw [impartial_def, neg_nim]
  /-
    case h
    o : Ordinal.{u_1}
    IH : ∀ (k : Ordinal.{u_1}), LT.lt k o → (SetTheory.PGame.nim k).Impartial
    ⊢ And (HasEquiv.Equiv (SetTheory.PGame.nim o) (SetTheory.PGame.nim o)) (And (∀ …
  -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  refine ⟨equiv_rfl, fun i => ?_, fun i => ?_⟩ <;> simpa using IH _ (typein_lt_self _)
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem nim_fuzzy_zero_of_ne_zero {o : Ordinal} (ho : o ≠ 0) : nim o ‖ 0 := by
  /-
    o : Ordinal.{u_1}
    ho : Ne o 0
    ⊢ (SetTheory.PGame.nim o).Fuzzy 0
  -/
  rw [Impartial.fuzzy_zero_iff_lf, lf_zero_le]
  /-
    o : Ordinal.{u_1}
    ho : Ne o 0
    ⊢ Exists fun j => LE.le ((SetTheory.PGame.nim o).moveRight j) 0
  -/
  use toRightMovesNim ⟨0, Ordinal.pos_iff_ne_zero.2 ho⟩
  /-
    case h
    o : Ordinal.{u_1}
    ho : Ne o 0
    ⊢ LE.le ((SetTheory.PGame.nim o).moveRight (SetTheory.PGame.toRightMovesNim ⟨0 …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem nim_add_equiv_zero_iff (o₁ o₂ : Ordinal) : (nim o₁ + nim o₂ ≈ 0) ↔ o₁ = o₂ := by
  /-
    o₁ o₂ : Ordinal.{u_1}
    ⊢ Iff (HasEquiv.Equiv (HAdd.hAdd (SetTheory.PGame.nim o₁) (SetTheory.PGame.nim …
  -/
  constructor
    /-
      case mp
      o₁ o₂ : Ordinal.{u_1}
      ⊢ HasEquiv.Equiv (HAdd.hAdd (SetTheory.PGame.nim o₁) (SetTheory.PGame.nim o₂)) …
    -/
  · refine not_imp_not.1 fun hne : _ ≠ _ => (Impartial.not_equiv_zero_iff (nim o₁ + nim o₂)).2 ?_
    /-
      case mp
      o₁ o₂ : Ordinal.{u_1}
      hne : Ne o₁ o₂
      ⊢ (HAdd.hAdd (SetTheory.PGame.nim o₁) (SetTheory.PGame.nim o₂)).Fuzzy 0
    -/
    wlog h : o₁ < o₂
      /-
        case mp.inr
        o₁ o₂ : Ordinal.{u_1}
        hne : Ne o₁ o₂
        this : ∀ (o₁ o₂ : Ordinal.{u_1}), Ne o₁ o₂ → LT.lt o₁ o₂ → (HAdd.hAdd (SetTheo …
        h : Not (LT.lt o₁ o₂)
        ⊢ (HAdd.hAdd (SetTheory.PGame.nim o₁) (SetTheory.PGame.nim o₂)).Fuzzy 0
      -/
    · exact (fuzzy_congr_left add_comm_equiv).1 (this _ _ hne.symm (hne.lt_or_lt.resolve_left h))
      /-
        🎉 no goals
      -/
    /-
      o₁ o₂ : Ordinal.{u_1}
      hne : Ne o₁ o₂
      h : LT.lt o₁ o₂
      ⊢ (HAdd.hAdd (SetTheory.PGame.nim o₁) (SetTheory.PGame.nim o₂)).Fuzzy 0
    -/
    rw [Impartial.fuzzy_zero_iff_gf, zero_lf_le]
    /-
      o₁ o₂ : Ordinal.{u_1}
      hne : Ne o₁ o₂
      h : LT.lt o₁ o₂
      ⊢ Exists fun i => LE.le 0 ((HAdd.hAdd (SetTheory.PGame.nim o₁) (SetTheory.PGam …
    -/
    use toLeftMovesAdd (Sum.inr <| toLeftMovesNim ⟨_, h⟩)
      /-
        case h
        o₁ o₂ : Ordinal.{u_1}
        hne : Ne o₁ o₂
        h : LT.lt o₁ o₂
        ⊢ LE.le 0 ((HAdd.hAdd (SetTheory.PGame.nim o₁) (SetTheory.PGame.nim o₂)).moveL …
      -/
    · simpa using (Impartial.add_self (nim o₁)).2
      /-
        🎉 no goals
      -/
    /-
      case mpr
      o₁ o₂ : Ordinal.{u_1}
      ⊢ Eq o₁ o₂ → HasEquiv.Equiv (HAdd.hAdd (SetTheory.PGame.nim o₁) (SetTheory.PGa …
    -/
  · rintro rfl
    /-
      case mpr
      o₁ : Ordinal.{u_1}
      ⊢ HasEquiv.Equiv (HAdd.hAdd (SetTheory.PGame.nim o₁) (SetTheory.PGame.nim o₁)) 0
    -/
    exact Impartial.add_self (nim o₁)
    /-
      🎉 no goals
    -/


@[simp]
theorem nim_add_fuzzy_zero_iff {o₁ o₂ : Ordinal} : nim o₁ + nim o₂ ‖ 0 ↔ o₁ ≠ o₂ := by
  /-
    o₁ o₂ : Ordinal.{u_1}
    ⊢ Iff ((HAdd.hAdd (SetTheory.PGame.nim o₁) (SetTheory.PGame.nim o₂)).Fuzzy 0)  …
  -/
  rw [iff_not_comm, Impartial.not_fuzzy_zero_iff, nim_add_equiv_zero_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem nim_equiv_iff_eq {o₁ o₂ : Ordinal} : (nim o₁ ≈ nim o₂) ↔ o₁ = o₂ := by
  /-
    o₁ o₂ : Ordinal.{u_1}
    ⊢ Iff (HasEquiv.Equiv (SetTheory.PGame.nim o₁) (SetTheory.PGame.nim o₂)) (Eq o …
  -/
  rw [Impartial.equiv_iff_add_equiv_zero, nim_add_equiv_zero_iff]
  /-
    🎉 no goals
  -/


/-- The Grundy value of an impartial game is recursively defined as the minimum excluded value
(the infimum of the complement) of the Grundy values of either its left or right options.

This is the ordinal which corresponds to the game of nim that the game is equivalent to.

This function takes a value in `Nimber`. This is a type synonym for the ordinals which has the same
ordering, but addition in `Nimber` is such that it corresponds to the grundy value of the addition
of games. See that file for more information on nimbers and their arithmetic. -/
noncomputable def grundyValue (G : PGame.{u}) : Nimber.{u} :=
  sInf (Set.range fun i => grundyValue (G.moveLeft i))ᶜ
termination_by G


theorem grundyValue_eq_sInf_moveLeft (G : PGame) :
    grundyValue G = sInf (Set.range (grundyValue ∘ G.moveLeft))ᶜ := by
  /-
    G : SetTheory.PGame
    ⊢ Eq G.grundyValue (InfSet.sInf (HasCompl.compl (Set.range (Function.comp SetT …
  -/
  rw [grundyValue]; rfl
                    /-
                      🎉 no goals
                    -/


set_option linter.deprecated false in
@[deprecated grundyValue_eq_sInf_moveLeft (since := "2024-09-16")]
theorem grundyValue_eq_mex_left (G : PGame) :
    grundyValue G = Ordinal.mex fun i => grundyValue (G.moveLeft i) :=
  grundyValue_eq_sInf_moveLeft G


theorem grundyValue_ne_moveLeft {G : PGame} (i : G.LeftMoves) :
    grundyValue (G.moveLeft i) ≠ grundyValue G := by
  /-
    G : SetTheory.PGame
    i : G.LeftMoves
    ⊢ Ne (G.moveLeft i).grundyValue G.grundyValue
  -/
  conv_rhs => rw [grundyValue_eq_sInf_moveLeft]
  have := csInf_mem (nonempty_of_not_bddAbove <|
    Nimber.not_bddAbove_compl_of_small (Set.range fun i => grundyValue (G.moveLeft i)))
  /-
    G : SetTheory.PGame
    i : G.LeftMoves
    this : Membership.mem (HasCompl.compl (Set.range fun i => (G.moveLeft i).grund …
    ⊢ Ne (G.moveLeft i).grundyValue (InfSet.sInf (HasCompl.compl (Set.range (Funct …
  -/
  rw [Set.mem_compl_iff, Set.mem_range, not_exists] at this
  /-
    G : SetTheory.PGame
    i : G.LeftMoves
    this : ∀ (x : G.LeftMoves), Not (Eq (G.moveLeft x).grundyValue (InfSet.sInf (H …
    ⊢ Ne (G.moveLeft i).grundyValue (InfSet.sInf (HasCompl.compl (Set.range (Funct …
  -/
  exact this _
  /-
    🎉 no goals
  -/


theorem le_grundyValue_of_Iio_subset_moveLeft {G : PGame} {o : Nimber}
    (h : Set.Iio o ⊆ Set.range (grundyValue ∘ G.moveLeft)) : o ≤ grundyValue G := by
  /-
    G : SetTheory.PGame
    o : Nimber
    h : HasSubset.Subset (Set.Iio o) (Set.range (Function.comp SetTheory.PGame.gru …
    ⊢ LE.le o G.grundyValue
  -/
  by_contra! ho
  /-
    G : SetTheory.PGame
    o : Nimber
    h : HasSubset.Subset (Set.Iio o) (Set.range (Function.comp SetTheory.PGame.gru …
    ho : LT.lt G.grundyValue o
    ⊢ False
  -/
  obtain ⟨i, hi⟩ := h ho
  /-
    case intro
    G : SetTheory.PGame
    o : Nimber
    h : HasSubset.Subset (Set.Iio o) (Set.range (Function.comp SetTheory.PGame.gru …
    ho : LT.lt G.grundyValue o
    i : G.LeftMoves
    hi : Eq (Function.comp SetTheory.PGame.grundyValue G.moveLeft i) G.grundyValue
    ⊢ False
  -/
  exact grundyValue_ne_moveLeft i hi
  /-
    🎉 no goals
  -/


theorem exists_grundyValue_moveLeft_of_lt {G : PGame} {o : Nimber} (h : o < grundyValue G) :
    ∃ i, grundyValue (G.moveLeft i) = o := by
  /-
    G : SetTheory.PGame
    o : Nimber
    h : LT.lt o G.grundyValue
    ⊢ Exists fun i => Eq (G.moveLeft i).grundyValue o
  -/
  rw [grundyValue_eq_sInf_moveLeft] at h
  /-
    G : SetTheory.PGame
    o : Nimber
    h : LT.lt o (InfSet.sInf (HasCompl.compl (Set.range (Function.comp SetTheory.P …
    ⊢ Exists fun i => Eq (G.moveLeft i).grundyValue o
  -/
  by_contra ha
  /-
    G : SetTheory.PGame
    o : Nimber
    h : LT.lt o (InfSet.sInf (HasCompl.compl (Set.range (Function.comp SetTheory.P …
    ha : Not (Exists fun i => Eq (G.moveLeft i).grundyValue o)
    ⊢ False
  -/
  exact h.not_le (csInf_le' ha)
  /-
    🎉 no goals
  -/


theorem grundyValue_le_of_forall_moveLeft {G : PGame} {o : Nimber}
    (h : ∀ i, grundyValue (G.moveLeft i) ≠ o) : G.grundyValue ≤ o := by
  /-
    G : SetTheory.PGame
    o : Nimber
    h : ∀ (i : G.LeftMoves), Ne (G.moveLeft i).grundyValue o
    ⊢ LE.le G.grundyValue o
  -/
  contrapose! h
  /-
    G : SetTheory.PGame
    o : Nimber
    h : LT.lt o G.grundyValue
    ⊢ Exists fun i => Eq (G.moveLeft i).grundyValue o
  -/
  exact exists_grundyValue_moveLeft_of_lt h
  /-
    🎉 no goals
  -/


/-- The **Sprague-Grundy theorem** states that every impartial game is equivalent to a game of nim,
namely the game of nim corresponding to the game's Grundy value. -/
theorem equiv_nim_grundyValue (G : PGame.{u}) [G.Impartial] :
    G ≈ nim (toOrdinal (grundyValue G)) := by
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    ⊢ HasEquiv.Equiv G (SetTheory.PGame.nim (Nimber.toOrdinal G.grundyValue))
  -/
  rw [Impartial.equiv_iff_add_equiv_zero, ← Impartial.forall_leftMoves_fuzzy_iff_equiv_zero]
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    ⊢ ∀ (i : (HAdd.hAdd G (SetTheory.PGame.nim (Nimber.toOrdinal G.grundyValue))). …
  -/
  intro x
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    x : (HAdd.hAdd G (SetTheory.PGame.nim (Nimber.toOrdinal G.grundyValue))).LeftM …
    ⊢ ((HAdd.hAdd G (SetTheory.PGame.nim (Nimber.toOrdinal G.grundyValue))).moveLe …
  -/
  apply leftMoves_add_cases x <;>
  /-
    case hl
    G : SetTheory.PGame
    inst✝ : G.Impartial
    x : (HAdd.hAdd G (SetTheory.PGame.nim (Nimber.toOrdinal G.grundyValue))).LeftM …
    ⊢ ∀ (i : G.LeftMoves), ((HAdd.hAdd G (SetTheory.PGame.nim (Nimber.toOrdinal G. …
  -/
  intro i
  · rw [add_moveLeft_inl,
      ← fuzzy_congr_left (add_congr_left (Equiv.symm (equiv_nim_grundyValue _))),
      nim_add_fuzzy_zero_iff]
    /-
      case hl
      G : SetTheory.PGame
      inst✝ : G.Impartial
      x : (HAdd.hAdd G (SetTheory.PGame.nim (Nimber.toOrdinal G.grundyValue))).LeftM …
      i : G.LeftMoves
      ⊢ Ne (Nimber.toOrdinal (G.moveLeft i).grundyValue) (Nimber.toOrdinal G.grundyV …
    -/
    exact grundyValue_ne_moveLeft i
    /-
      🎉 no goals
    -/
    /-
      case hr
      G : SetTheory.PGame
      inst✝ : G.Impartial
      x : (HAdd.hAdd G (SetTheory.PGame.nim (Nimber.toOrdinal G.grundyValue))).LeftM …
      i : (SetTheory.PGame.nim (Nimber.toOrdinal G.grundyValue)).LeftMoves
      ⊢ ((HAdd.hAdd G (SetTheory.PGame.nim (Nimber.toOrdinal G.grundyValue))).moveLe …
    -/
  · rw [add_moveLeft_inr, ← Impartial.exists_left_move_equiv_iff_fuzzy_zero]
    /-
      case hr
      G : SetTheory.PGame
      inst✝ : G.Impartial
      x : (HAdd.hAdd G (SetTheory.PGame.nim (Nimber.toOrdinal G.grundyValue))).LeftM …
      i : (SetTheory.PGame.nim (Nimber.toOrdinal G.grundyValue)).LeftMoves
      ⊢ Exists fun i_1 => HasEquiv.Equiv ((HAdd.hAdd G ((SetTheory.PGame.nim (Nimber …
    -/
    obtain ⟨j, hj⟩ := exists_grundyValue_moveLeft_of_lt <| toLeftMovesNim_symm_lt i
    /-
      case hr.intro
      G : SetTheory.PGame
      inst✝ : G.Impartial
      x : (HAdd.hAdd G (SetTheory.PGame.nim (Nimber.toOrdinal G.grundyValue))).LeftM …
      i : (SetTheory.PGame.nim (Nimber.toOrdinal G.grundyValue)).LeftMoves
      j : G.LeftMoves
      hj : Eq (G.moveLeft j).grundyValue ↑(SetTheory.PGame.toLeftMovesNim.symm i)
      ⊢ Exists fun i_1 => HasEquiv.Equiv ((HAdd.hAdd G ((SetTheory.PGame.nim (Nimber …
    -/
    use toLeftMovesAdd (Sum.inl j)
    /-
      case h
      G : SetTheory.PGame
      inst✝ : G.Impartial
      x : (HAdd.hAdd G (SetTheory.PGame.nim (Nimber.toOrdinal G.grundyValue))).LeftM …
      i : (SetTheory.PGame.nim (Nimber.toOrdinal G.grundyValue)).LeftMoves
      j : G.LeftMoves
      hj : Eq (G.moveLeft j).grundyValue ↑(SetTheory.PGame.toLeftMovesNim.symm i)
      ⊢ HasEquiv.Equiv ((HAdd.hAdd G ((SetTheory.PGame.nim (Nimber.toOrdinal G.grund …
    -/
    rw [add_moveLeft_inl, moveLeft_nim]
    /-
      case h
      G : SetTheory.PGame
      inst✝ : G.Impartial
      x : (HAdd.hAdd G (SetTheory.PGame.nim (Nimber.toOrdinal G.grundyValue))).LeftM …
      i : (SetTheory.PGame.nim (Nimber.toOrdinal G.grundyValue)).LeftMoves
      j : G.LeftMoves
      hj : Eq (G.moveLeft j).grundyValue ↑(SetTheory.PGame.toLeftMovesNim.symm i)
      ⊢ HasEquiv.Equiv (HAdd.hAdd (G.moveLeft j) (SetTheory.PGame.nim ↑(SetTheory.PG …
    -/
    exact Equiv.trans (add_congr_left (equiv_nim_grundyValue _)) (hj ▸ Impartial.add_self _)
    /-
      🎉 no goals
    -/
termination_by G


theorem grundyValue_eq_iff_equiv_nim {G : PGame} [G.Impartial] {o : Nimber} :
    grundyValue G = o ↔ G ≈ nim (toOrdinal o) :=
      /-
        G : SetTheory.PGame
        inst✝ : G.Impartial
        o : Nimber
        ⊢ Eq G.grundyValue o → HasEquiv.Equiv G (SetTheory.PGame.nim (Nimber.toOrdinal …
      -/
  ⟨by rintro rfl; exact equiv_nim_grundyValue G,
                  /-
                    🎉 no goals
                  -/
      /-
        G : SetTheory.PGame
        inst✝ : G.Impartial
        o : Nimber
        ⊢ HasEquiv.Equiv G (SetTheory.PGame.nim (Nimber.toOrdinal o)) → Eq G.grundyVal …
      -/
   by intro h; rw [← nim_equiv_iff_eq]; exact Equiv.trans (Equiv.symm (equiv_nim_grundyValue G)) h⟩
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem nim_grundyValue (o : Ordinal.{u}) : grundyValue (nim o) = ∗o :=
  grundyValue_eq_iff_equiv_nim.2 PGame.equiv_rfl


theorem grundyValue_eq_iff_equiv (G H : PGame) [G.Impartial] [H.Impartial] :
    grundyValue G = grundyValue H ↔ (G ≈ H) :=
  grundyValue_eq_iff_equiv_nim.trans (equiv_congr_left.1 (equiv_nim_grundyValue H) _).symm


@[simp]
theorem grundyValue_zero : grundyValue 0 = 0 :=
  grundyValue_eq_iff_equiv_nim.2 (Equiv.symm nim_zero_equiv)


theorem grundyValue_iff_equiv_zero (G : PGame) [G.Impartial] : grundyValue G = 0 ↔ G ≈ 0 := by
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    ⊢ Iff (Eq G.grundyValue 0) (HasEquiv.Equiv G 0)
  -/
  rw [← grundyValue_eq_iff_equiv, grundyValue_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem grundyValue_star : grundyValue star = 1 :=
  grundyValue_eq_iff_equiv_nim.2 (Equiv.symm nim_one_equiv)


@[simp]
theorem grundyValue_neg (G : PGame) [G.Impartial] : grundyValue (-G) = grundyValue G := by
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    ⊢ Eq (Neg.neg G).grundyValue G.grundyValue
  -/
  rw [grundyValue_eq_iff_equiv_nim, neg_equiv_iff, neg_nim, ← grundyValue_eq_iff_equiv_nim]
  /-
    🎉 no goals
  -/


theorem grundyValue_eq_sInf_moveRight (G : PGame) [G.Impartial] :
    grundyValue G = sInf (Set.range (grundyValue ∘ G.moveRight))ᶜ := by
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    ⊢ Eq G.grundyValue (InfSet.sInf (HasCompl.compl (Set.range (Function.comp SetT …
  -/
  obtain ⟨l, r, L, R⟩ := G
  /-
    case mk
    l r : Type u_1
    L : l → SetTheory.PGame
    R : r → SetTheory.PGame
    inst✝ : (SetTheory.PGame.mk l r L R).Impartial
    ⊢ Eq (SetTheory.PGame.mk l r L R).grundyValue (InfSet.sInf (HasCompl.compl (Se …
  -/
  rw [← grundyValue_neg, grundyValue_eq_sInf_moveLeft]
  /-
    case mk
    l r : Type u_1
    L : l → SetTheory.PGame
    R : r → SetTheory.PGame
    inst✝ : (SetTheory.PGame.mk l r L R).Impartial
    ⊢ Eq (InfSet.sInf (HasCompl.compl (Set.range (Function.comp SetTheory.PGame.gr …
  -/
  iterate 3 apply congr_arg
  /-
    case mk.h.h.h
    l r : Type u_1
    L : l → SetTheory.PGame
    R : r → SetTheory.PGame
    inst✝ : (SetTheory.PGame.mk l r L R).Impartial
    ⊢ Eq (Function.comp SetTheory.PGame.grundyValue (Neg.neg (SetTheory.PGame.mk l …
  -/
  ext i
  /-
    case mk.h.h.h.h
    l r : Type u_1
    L : l → SetTheory.PGame
    R : r → SetTheory.PGame
    inst✝ : (SetTheory.PGame.mk l r L R).Impartial
    i : (Neg.neg (SetTheory.PGame.mk l r L R)).LeftMoves
    ⊢ Eq (Function.comp SetTheory.PGame.grundyValue (Neg.neg (SetTheory.PGame.mk l …
  -/
  exact @grundyValue_neg _ (@Impartial.moveRight_impartial ⟨l, r, L, R⟩ _ _)
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated grundyValue_eq_sInf_moveRight (since := "2024-09-16")]
theorem grundyValue_eq_mex_right (G : PGame) [G.Impartial] :
    grundyValue G = Ordinal.mex.{u, u} fun i => grundyValue (G.moveRight i) :=
  grundyValue_eq_sInf_moveRight G


theorem grundyValue_ne_moveRight {G : PGame} [G.Impartial] (i : G.RightMoves) :
    grundyValue (G.moveRight i) ≠ grundyValue G := by
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    i : G.RightMoves
    ⊢ Ne (G.moveRight i).grundyValue G.grundyValue
  -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  convert grundyValue_ne_moveLeft (toLeftMovesNeg i) using 1 <;> simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem le_grundyValue_of_Iio_subset_moveRight {G : PGame} [G.Impartial] {o : Nimber}
    (h : Set.Iio o ⊆ Set.range (grundyValue ∘ G.moveRight)) : o ≤ grundyValue G := by
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    o : Nimber
    h : HasSubset.Subset (Set.Iio o) (Set.range (Function.comp SetTheory.PGame.gru …
    ⊢ LE.le o G.grundyValue
  -/
  by_contra! ho
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    o : Nimber
    h : HasSubset.Subset (Set.Iio o) (Set.range (Function.comp SetTheory.PGame.gru …
    ho : LT.lt G.grundyValue o
    ⊢ False
  -/
  obtain ⟨i, hi⟩ := h ho
  /-
    case intro
    G : SetTheory.PGame
    inst✝ : G.Impartial
    o : Nimber
    h : HasSubset.Subset (Set.Iio o) (Set.range (Function.comp SetTheory.PGame.gru …
    ho : LT.lt G.grundyValue o
    i : G.RightMoves
    hi : Eq (Function.comp SetTheory.PGame.grundyValue G.moveRight i) G.grundyValue
    ⊢ False
  -/
  exact grundyValue_ne_moveRight i hi
  /-
    🎉 no goals
  -/


theorem exists_grundyValue_moveRight_of_lt {G : PGame} [G.Impartial] {o : Nimber}
    (h : o < grundyValue G) : ∃ i, grundyValue (G.moveRight i) = o := by
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    o : Nimber
    h : LT.lt o G.grundyValue
    ⊢ Exists fun i => Eq (G.moveRight i).grundyValue o
  -/
  rw [← grundyValue_neg] at h
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    o : Nimber
    h : LT.lt o (Neg.neg G).grundyValue
    ⊢ Exists fun i => Eq (G.moveRight i).grundyValue o
  -/
  obtain ⟨i, hi⟩ := exists_grundyValue_moveLeft_of_lt h
  /-
    case intro
    G : SetTheory.PGame
    inst✝ : G.Impartial
    o : Nimber
    h : LT.lt o (Neg.neg G).grundyValue
    i : (Neg.neg G).LeftMoves
    hi : Eq ((Neg.neg G).moveLeft i).grundyValue o
    ⊢ Exists fun i => Eq (G.moveRight i).grundyValue o
  -/
  use toLeftMovesNeg.symm i
  /-
    case h
    G : SetTheory.PGame
    inst✝ : G.Impartial
    o : Nimber
    h : LT.lt o (Neg.neg G).grundyValue
    i : (Neg.neg G).LeftMoves
    hi : Eq ((Neg.neg G).moveLeft i).grundyValue o
    ⊢ Eq (G.moveRight (SetTheory.PGame.toLeftMovesNeg.symm i)).grundyValue o
  -/
  rwa [← grundyValue_neg, ← moveLeft_neg]
  /-
    🎉 no goals
  -/


theorem grundyValue_le_of_forall_moveRight {G : PGame} [G.Impartial] {o : Nimber}
    (h : ∀ i, grundyValue (G.moveRight i) ≠ o) : G.grundyValue ≤ o := by
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    o : Nimber
    h : ∀ (i : G.RightMoves), Ne (G.moveRight i).grundyValue o
    ⊢ LE.le G.grundyValue o
  -/
  contrapose! h
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    o : Nimber
    h : LT.lt o G.grundyValue
    ⊢ Exists fun i => Eq (G.moveRight i).grundyValue o
  -/
  exact exists_grundyValue_moveRight_of_lt h
  /-
    🎉 no goals
  -/


/-- The Grundy value of the sum of two nim games equals their nimber addition. -/
theorem grundyValue_nim_add_nim (x y : Ordinal) : grundyValue (nim x + nim y) = ∗x + ∗y := by
  /-
    x y : Ordinal.{u_1}
    ⊢ Eq (HAdd.hAdd (SetTheory.PGame.nim x) (SetTheory.PGame.nim y)).grundyValue ( …
  -/
  apply (grundyValue_le_of_forall_moveLeft _).antisymm (le_grundyValue_of_Iio_subset_moveLeft _)
    /-
      x y : Ordinal.{u_1}
      ⊢ ∀ (i : (HAdd.hAdd (SetTheory.PGame.nim x) (SetTheory.PGame.nim y)).LeftMoves …
    -/
  · intro i
    /-
      x y : Ordinal.{u_1}
      i : (HAdd.hAdd (SetTheory.PGame.nim x) (SetTheory.PGame.nim y)).LeftMoves
      ⊢ Ne ((HAdd.hAdd (SetTheory.PGame.nim x) (SetTheory.PGame.nim y)).moveLeft i). …
    -/
    apply leftMoves_add_cases i <;> intro j <;> have := (toLeftMovesNim_symm_lt j).ne
      /-
        case hl
        x y : Ordinal.{u_1}
        i : (HAdd.hAdd (SetTheory.PGame.nim x) (SetTheory.PGame.nim y)).LeftMoves
        j : (SetTheory.PGame.nim x).LeftMoves
        this : Ne (↑(SetTheory.PGame.toLeftMovesNim.symm j)) x
        ⊢ Ne ((HAdd.hAdd (SetTheory.PGame.nim x) (SetTheory.PGame.nim y)).moveLeft (Se …
      -/
    · simpa [grundyValue_nim_add_nim (toLeftMovesNim.symm j) y]
      /-
        🎉 no goals
      -/
      /-
        case hr
        x y : Ordinal.{u_1}
        i : (HAdd.hAdd (SetTheory.PGame.nim x) (SetTheory.PGame.nim y)).LeftMoves
        j : (SetTheory.PGame.nim y).LeftMoves
        this : Ne (↑(SetTheory.PGame.toLeftMovesNim.symm j)) y
        ⊢ Ne ((HAdd.hAdd (SetTheory.PGame.nim x) (SetTheory.PGame.nim y)).moveLeft (Se …
      -/
    · simpa [grundyValue_nim_add_nim x (toLeftMovesNim.symm j)]
      /-
        🎉 no goals
      -/
    /-
      x y : Ordinal.{u_1}
      ⊢ HasSubset.Subset (Set.Iio (HAdd.hAdd (Ordinal.toNimber x) (Ordinal.toNimber  …
    -/
  · intro k hk
    /-
      x y : Ordinal.{u_1}
      k : Nimber
      hk : Membership.mem (Set.Iio (HAdd.hAdd (Ordinal.toNimber x) (Ordinal.toNimber …
      ⊢ Membership.mem (Set.range (Function.comp SetTheory.PGame.grundyValue (HAdd.h …
    -/
    obtain h | h := Nimber.lt_add_cases hk
      /-
        case inl
        x y : Ordinal.{u_1}
        k : Nimber
        hk : Membership.mem (Set.Iio (HAdd.hAdd (Ordinal.toNimber x) (Ordinal.toNimber …
        h : LT.lt (HAdd.hAdd k (Ordinal.toNimber y)) (Ordinal.toNimber x)
        ⊢ Membership.mem (Set.range (Function.comp SetTheory.PGame.grundyValue (HAdd.h …
      -/
    · let a := toOrdinal (k + ∗y)
      /-
        case inl
        x y : Ordinal.{u_1}
        k : Nimber
        hk : Membership.mem (Set.Iio (HAdd.hAdd (Ordinal.toNimber x) (Ordinal.toNimber …
        h : LT.lt (HAdd.hAdd k (Ordinal.toNimber y)) (Ordinal.toNimber x)
        a : Ordinal.{u_1} := Nimber.toOrdinal (HAdd.hAdd k (Ordinal.toNimber y))
        ⊢ Membership.mem (Set.range (Function.comp SetTheory.PGame.grundyValue (HAdd.h …
      -/
      use toLeftMovesAdd (Sum.inl (toLeftMovesNim ⟨a, h⟩))
      /-
        case h
        x y : Ordinal.{u_1}
        k : Nimber
        hk : Membership.mem (Set.Iio (HAdd.hAdd (Ordinal.toNimber x) (Ordinal.toNimber …
        h : LT.lt (HAdd.hAdd k (Ordinal.toNimber y)) (Ordinal.toNimber x)
        a : Ordinal.{u_1} := Nimber.toOrdinal (HAdd.hAdd k (Ordinal.toNimber y))
        ⊢ Eq (Function.comp SetTheory.PGame.grundyValue (HAdd.hAdd (SetTheory.PGame.ni …
      -/
      simp [a, grundyValue_nim_add_nim a y]
      /-
        🎉 no goals
      -/
      /-
        case inr
        x y : Ordinal.{u_1}
        k : Nimber
        hk : Membership.mem (Set.Iio (HAdd.hAdd (Ordinal.toNimber x) (Ordinal.toNimber …
        h : LT.lt (HAdd.hAdd k (Ordinal.toNimber x)) (Ordinal.toNimber y)
        ⊢ Membership.mem (Set.range (Function.comp SetTheory.PGame.grundyValue (HAdd.h …
      -/
    · let a := toOrdinal (k + ∗x)
      /-
        case inr
        x y : Ordinal.{u_1}
        k : Nimber
        hk : Membership.mem (Set.Iio (HAdd.hAdd (Ordinal.toNimber x) (Ordinal.toNimber …
        h : LT.lt (HAdd.hAdd k (Ordinal.toNimber x)) (Ordinal.toNimber y)
        a : Ordinal.{u_1} := Nimber.toOrdinal (HAdd.hAdd k (Ordinal.toNimber x))
        ⊢ Membership.mem (Set.range (Function.comp SetTheory.PGame.grundyValue (HAdd.h …
      -/
      use toLeftMovesAdd (Sum.inr (toLeftMovesNim ⟨a, h⟩))
      /-
        case h
        x y : Ordinal.{u_1}
        k : Nimber
        hk : Membership.mem (Set.Iio (HAdd.hAdd (Ordinal.toNimber x) (Ordinal.toNimber …
        h : LT.lt (HAdd.hAdd k (Ordinal.toNimber x)) (Ordinal.toNimber y)
        a : Ordinal.{u_1} := Nimber.toOrdinal (HAdd.hAdd k (Ordinal.toNimber x))
        ⊢ Eq (Function.comp SetTheory.PGame.grundyValue (HAdd.hAdd (SetTheory.PGame.ni …
      -/
      simp [a, grundyValue_nim_add_nim x a, add_comm (∗x)]
      /-
        🎉 no goals
      -/
termination_by (x, y)


theorem nim_add_nim_equiv (x y : Ordinal) :
    nim x + nim y ≈ nim (toOrdinal (∗x + ∗y)) := by
  /-
    x y : Ordinal.{u_1}
    ⊢ HasEquiv.Equiv (HAdd.hAdd (SetTheory.PGame.nim x) (SetTheory.PGame.nim y)) ( …
  -/
  rw [← grundyValue_eq_iff_equiv_nim, grundyValue_nim_add_nim]
  /-
    🎉 no goals
  -/


@[simp]
theorem grundyValue_add (G H : PGame) [G.Impartial] [H.Impartial] :
    grundyValue (G + H) = grundyValue G + grundyValue H := by
  rw [← (grundyValue G).toOrdinal_toNimber, ← (grundyValue H).toOrdinal_toNimber,
    ← grundyValue_nim_add_nim, grundyValue_eq_iff_equiv]
  /-
    G H : SetTheory.PGame
    inst✝¹ : G.Impartial
    inst✝ : H.Impartial
    ⊢ HasEquiv.Equiv (HAdd.hAdd G H) (HAdd.hAdd (SetTheory.PGame.nim (Nimber.toOrd …
  -/
  exact add_congr (equiv_nim_grundyValue G) (equiv_nim_grundyValue H)
  /-
    🎉 no goals
  -/



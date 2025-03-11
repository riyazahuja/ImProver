/-- The type of pre-games, before we have quotiented
  by equivalence (`PGame.Setoid`). In ZFC, a combinatorial game is constructed from
  two sets of combinatorial games that have been constructed at an earlier
  stage. To do this in type theory, we say that a pre-game is built
  inductively from two families of pre-games indexed over any type
  in Type u. The resulting type `PGame.{u}` lives in `Type (u+1)`,
  reflecting that it is a proper class in ZFC. -/
inductive PGame : Type (u + 1)
  | mk : ∀ α β : Type u, (α → PGame) → (β → PGame) → PGame

compile_inductive% PGame


/-- The indexing type for allowable moves by Left. -/
def LeftMoves : PGame → Type u
  | mk l _ _ _ => l


/-- The indexing type for allowable moves by Right. -/
def RightMoves : PGame → Type u
  | mk _ r _ _ => r


/-- The new game after Left makes an allowed move. -/
def moveLeft : ∀ g : PGame, LeftMoves g → PGame
  | mk _l _ L _ => L


/-- The new game after Right makes an allowed move. -/
def moveRight : ∀ g : PGame, RightMoves g → PGame
  | mk _ _r _ R => R


@[simp]
theorem leftMoves_mk {xl xr xL xR} : (⟨xl, xr, xL, xR⟩ : PGame).LeftMoves = xl :=
  rfl


@[simp]
theorem moveLeft_mk {xl xr xL xR} : (⟨xl, xr, xL, xR⟩ : PGame).moveLeft = xL :=
  rfl


@[simp]
theorem rightMoves_mk {xl xr xL xR} : (⟨xl, xr, xL, xR⟩ : PGame).RightMoves = xr :=
  rfl


@[simp]
theorem moveRight_mk {xl xr xL xR} : (⟨xl, xr, xL, xR⟩ : PGame).moveRight = xR :=
  rfl

-- TODO define this at the level of games, as well, and perhaps also for finsets of games.

/-- Construct a pre-game from list of pre-games describing the available moves for Left and Right.
-/
def ofLists (L R : List PGame.{u}) : PGame.{u} :=
  mk (ULift (Fin L.length)) (ULift (Fin R.length)) (fun i => L[i.down.1]) fun j ↦ R[j.down.1]


theorem leftMoves_ofLists (L R : List PGame) : (ofLists L R).LeftMoves = ULift (Fin L.length) :=
  rfl


theorem rightMoves_ofLists (L R : List PGame) : (ofLists L R).RightMoves = ULift (Fin R.length) :=
  rfl


/-- Converts a number into a left move for `ofLists`.

This is just an abbreviation for `Equiv.ulift.symm` -/
abbrev toOfListsLeftMoves {L R : List PGame} : Fin L.length ≃ (ofLists L R).LeftMoves :=
  Equiv.ulift.symm


/-- Converts a number into a right move for `ofLists`.

This is just an abbreviation for `Equiv.ulift.symm` -/
abbrev toOfListsRightMoves {L R : List PGame} : Fin R.length ≃ (ofLists L R).RightMoves :=
  Equiv.ulift.symm


@[simp]
theorem ofLists_moveLeft' {L R : List PGame} (i : (ofLists L R).LeftMoves) :
    (ofLists L R).moveLeft i = L[i.down.val] :=
  rfl


theorem ofLists_moveLeft {L R : List PGame} (i : Fin L.length) :
    (ofLists L R).moveLeft (ULift.up i) = L[i] :=
  rfl


@[simp]
theorem ofLists_moveRight' {L R : List PGame} (i : (ofLists L R).RightMoves) :
    (ofLists L R).moveRight i = R[i.down.val] :=
  rfl


theorem ofLists_moveRight {L R : List PGame} (i : Fin R.length) :
    (ofLists L R).moveRight (ULift.up i) = R[i] :=
  rfl


/-- A variant of `PGame.recOn` expressed in terms of `PGame.moveLeft` and `PGame.moveRight`.

Both this and `PGame.recOn` describe Conway induction on games. -/
@[elab_as_elim]
def moveRecOn {C : PGame → Sort*} (x : PGame)
    (IH : ∀ y : PGame, (∀ i, C (y.moveLeft i)) → (∀ j, C (y.moveRight j)) → C y) : C x :=
  x.recOn fun yl yr yL yR => IH (mk yl yr yL yR)


/-- `IsOption x y` means that `x` is either a left or right option for `y`. -/
@[mk_iff]
inductive IsOption : PGame → PGame → Prop
  | moveLeft {x : PGame} (i : x.LeftMoves) : IsOption (x.moveLeft i) x
  | moveRight {x : PGame} (i : x.RightMoves) : IsOption (x.moveRight i) x


theorem IsOption.mk_left {xl xr : Type u} (xL : xl → PGame) (xR : xr → PGame) (i : xl) :
    (xL i).IsOption (mk xl xr xL xR) :=
  @IsOption.moveLeft (mk _ _ _ _) i


theorem IsOption.mk_right {xl xr : Type u} (xL : xl → PGame) (xR : xr → PGame) (i : xr) :
    (xR i).IsOption (mk xl xr xL xR) :=
  @IsOption.moveRight (mk _ _ _ _) i


theorem wf_isOption : WellFounded IsOption :=
  ⟨fun x =>
    moveRecOn x fun x IHl IHr =>
      Acc.intro x fun y h => by
        induction h with
        | moveLeft i => exact IHl i
        | moveRight j => exact IHr j⟩


/-- `Subsequent x y` says that `x` can be obtained by playing some nonempty sequence of moves from
`y`. It is the transitive closure of `IsOption`. -/
def Subsequent : PGame → PGame → Prop :=
  TransGen IsOption


instance : IsTrans _ Subsequent :=
  inferInstanceAs <| IsTrans _ (TransGen _)


@[trans]
theorem Subsequent.trans {x y z} : Subsequent x y → Subsequent y z → Subsequent x z :=
  TransGen.trans


theorem wf_subsequent : WellFounded Subsequent :=
  wf_isOption.transGen


instance : WellFoundedRelation PGame :=
  ⟨_, wf_subsequent⟩


@[simp]
theorem Subsequent.moveLeft {x : PGame} (i : x.LeftMoves) : Subsequent (x.moveLeft i) x :=
  TransGen.single (IsOption.moveLeft i)


@[simp]
theorem Subsequent.moveRight {x : PGame} (j : x.RightMoves) : Subsequent (x.moveRight j) x :=
  TransGen.single (IsOption.moveRight j)


@[simp]
theorem Subsequent.mk_left {xl xr} (xL : xl → PGame) (xR : xr → PGame) (i : xl) :
    Subsequent (xL i) (mk xl xr xL xR) :=
  @Subsequent.moveLeft (mk _ _ _ _) i


@[simp]
theorem Subsequent.mk_right {xl xr} (xL : xl → PGame) (xR : xr → PGame) (j : xr) :
    Subsequent (xR j) (mk xl xr xL xR) :=
  @Subsequent.moveRight (mk _ _ _ _) j


/--
Discharges proof obligations of the form `⊢ Subsequent ..` arising in termination proofs
of definitions using well-founded recursion on `PGame`.
-/
macro "pgame_wf_tac" : tactic =>
  `(tactic| solve_by_elim (config := { maxDepth := 8 })
    [Prod.Lex.left, Prod.Lex.right, PSigma.Lex.left, PSigma.Lex.right,
    Subsequent.moveLeft, Subsequent.moveRight, Subsequent.mk_left, Subsequent.mk_right,
    Subsequent.trans] )

-- Register some consequences of pgame_wf_tac as simp-lemmas for convenience
-- (which are applied by default for WF goals)


@[simp]
theorem Subsequent.mk_right' (xL : xl → PGame) (xR : xr → PGame) (j : RightMoves (mk xl xr xL xR)) :
    Subsequent (xR j) (mk xl xr xL xR) := by
  /-
    xl xr : Type u
    xL : xl → SetTheory.PGame
    xR : xr → SetTheory.PGame
    j : (SetTheory.PGame.mk xl xr xL xR).RightMoves
    ⊢ (xR j).Subsequent (SetTheory.PGame.mk xl xr xL xR)
  -/
  pgame_wf_tac
  /-
    🎉 no goals
  -/


@[simp] theorem Subsequent.moveRight_mk_left {xR : xr → PGame} {i : xl} (xL : xl → PGame) (j) :
    Subsequent ((xL i).moveRight j) (mk xl xr xL xR) := by
  /-
    xl xr : Type u
    xR : xr → SetTheory.PGame
    i : xl
    xL : xl → SetTheory.PGame
    j : (xL i).RightMoves
    ⊢ ((xL i).moveRight j).Subsequent (SetTheory.PGame.mk xl xr xL xR)
  -/
  pgame_wf_tac
  /-
    🎉 no goals
  -/


@[simp] theorem Subsequent.moveRight_mk_right {xL : xl → PGame} {i : xr} (xR : xr → PGame) (j) :
    Subsequent ((xR i).moveRight j) (mk xl xr xL xR) := by
  /-
    xl xr : Type u
    xL : xl → SetTheory.PGame
    i : xr
    xR : xr → SetTheory.PGame
    j : (xR i).RightMoves
    ⊢ ((xR i).moveRight j).Subsequent (SetTheory.PGame.mk xl xr xL xR)
  -/
  pgame_wf_tac
  /-
    🎉 no goals
  -/


@[simp] theorem Subsequent.moveLeft_mk_left {xR : xr → PGame} {i : xl} (xL : xl → PGame) (j) :
    Subsequent ((xL i).moveLeft j) (mk xl xr xL xR) := by
  /-
    xl xr : Type u
    xR : xr → SetTheory.PGame
    i : xl
    xL : xl → SetTheory.PGame
    j : (xL i).LeftMoves
    ⊢ ((xL i).moveLeft j).Subsequent (SetTheory.PGame.mk xl xr xL xR)
  -/
  pgame_wf_tac
  /-
    🎉 no goals
  -/


@[simp] theorem Subsequent.moveLeft_mk_right {xL : xl → PGame} {i : xr} (xR : xr → PGame) (j) :
    Subsequent ((xR i).moveLeft j) (mk xl xr xL xR) := by
  /-
    xl xr : Type u
    xL : xl → SetTheory.PGame
    i : xr
    xR : xr → SetTheory.PGame
    j : (xR i).LeftMoves
    ⊢ ((xR i).moveLeft j).Subsequent (SetTheory.PGame.mk xl xr xL xR)
  -/
  pgame_wf_tac
  /-
    🎉 no goals
  -/


/-- The pre-game `Zero` is defined by `0 = { | }`. -/
instance : Zero PGame :=
  ⟨⟨PEmpty, PEmpty, PEmpty.elim, PEmpty.elim⟩⟩


@[simp]
theorem zero_leftMoves : LeftMoves 0 = PEmpty :=
  rfl


@[simp]
theorem zero_rightMoves : RightMoves 0 = PEmpty :=
  rfl


instance isEmpty_zero_leftMoves : IsEmpty (LeftMoves 0) :=
  PEmpty.instIsEmpty


instance isEmpty_zero_rightMoves : IsEmpty (RightMoves 0) :=
  PEmpty.instIsEmpty


instance : Inhabited PGame :=
  ⟨0⟩


/-- The pre-game `One` is defined by `1 = { 0 | }`. -/
instance instOnePGame : One PGame :=
  ⟨⟨PUnit, PEmpty, fun _ => 0, PEmpty.elim⟩⟩


@[simp]
theorem one_leftMoves : LeftMoves 1 = PUnit :=
  rfl


@[simp]
theorem one_moveLeft (x) : moveLeft 1 x = 0 :=
  rfl


@[simp]
theorem one_rightMoves : RightMoves 1 = PEmpty :=
  rfl


instance uniqueOneLeftMoves : Unique (LeftMoves 1) :=
  PUnit.instUnique


instance isEmpty_one_rightMoves : IsEmpty (RightMoves 1) :=
  PEmpty.instIsEmpty


/-- Two pre-games are identical if their left and right sets are identical.
That is, `Identical x y` if every left move of `x` is identical to some left move of `y`,
every right move of `x` is identical to some right move of `y`, and vice versa. -/
def Identical : PGame.{u} → PGame.{u} → Prop
  | mk _ _ xL xR, mk _ _ yL yR =>
    Relator.BiTotal (fun i j ↦ Identical (xL i) (yL j)) ∧
      Relator.BiTotal (fun i j ↦ Identical (xR i) (yR j))


@[inherit_doc] scoped infix:50 " ≡ " => PGame.Identical


theorem identical_iff : ∀ {x y : PGame}, x ≡ y ↔
    Relator.BiTotal (x.moveLeft · ≡ y.moveLeft ·) ∧ Relator.BiTotal (x.moveRight · ≡ y.moveRight ·)
  | mk _ _ _ _, mk _ _ _ _ => Iff.rfl


@[refl, simp] protected theorem Identical.refl (x) : x ≡ x :=
  PGame.recOn x fun _ _ _ _ IHL IHR ↦ ⟨Relator.BiTotal.refl IHL, Relator.BiTotal.refl IHR⟩


protected theorem Identical.rfl {x} : x ≡ x := Identical.refl x


@[symm] protected theorem Identical.symm : ∀ {x y}, x ≡ y → y ≡ x
  | mk _ _ _ _, mk _ _ _ _, ⟨hL, hR⟩ => ⟨hL.symm fun _ _ h ↦ h.symm, hR.symm fun _ _ h ↦ h.symm⟩


theorem identical_comm {x y} : x ≡ y ↔ y ≡ x :=
  ⟨.symm, .symm⟩


@[trans] protected theorem Identical.trans : ∀ {x y z}, x ≡ y → y ≡ z → x ≡ z
  | mk _ _ _ _, mk _ _ _ _, mk _ _ _ _, ⟨hL₁, hR₁⟩, ⟨hL₂, hR₂⟩ =>
    ⟨hL₁.trans (fun _ _ _ h₁ h₂ ↦ h₁.trans h₂) hL₂, hR₁.trans (fun _ _ _ h₁ h₂ ↦ h₁.trans h₂) hR₂⟩


/-- `x ∈ₗ y` if `x` is identical to some left move of `y`. -/
def memₗ (x y : PGame.{u}) : Prop := ∃ b, x ≡ y.moveLeft b


/-- `x ∈ᵣ y` if `x` is identical to some right move of `y`. -/
def memᵣ (x y : PGame.{u}) : Prop := ∃ b, x ≡ y.moveRight b


@[inherit_doc] scoped infix:50 " ∈ₗ " => PGame.memₗ

@[inherit_doc] scoped infix:50 " ∈ᵣ " => PGame.memᵣ

@[inherit_doc PGame.memₗ] binder_predicate x " ∈ₗ " y:term => `($x ∈ₗ $y)

@[inherit_doc PGame.memᵣ] binder_predicate x " ∈ᵣ " y:term => `($x ∈ᵣ $y)


theorem memₗ_def {x y : PGame} : x ∈ₗ y ↔ ∃ b, x ≡ y.moveLeft b := .rfl

theorem memᵣ_def {x y : PGame} : x ∈ᵣ y ↔ ∃ b, x ≡ y.moveRight b := .rfl

theorem moveLeft_memₗ (x : PGame) (b) : x.moveLeft b ∈ₗ x := ⟨_, .rfl⟩

theorem moveRight_memᵣ (x : PGame) (b) : x.moveRight b ∈ᵣ x := ⟨_, .rfl⟩


theorem identical_of_isEmpty (x y : PGame)
    [IsEmpty x.LeftMoves] [IsEmpty x.RightMoves]
    [IsEmpty y.LeftMoves] [IsEmpty y.RightMoves] : x ≡ y :=
                      /-
                        x y : SetTheory.PGame
                        inst✝³ : IsEmpty x.LeftMoves
                        inst✝² : IsEmpty x.RightMoves
                        inst✝¹ : IsEmpty y.LeftMoves
                        inst✝ : IsEmpty y.RightMoves
                        ⊢ And (Relator.BiTotal fun x1 x2 => (x.moveLeft x1).Identical (y.moveLeft x2)) …
                      -/
  identical_iff.2 (by simp [biTotal_empty])
                      /-
                        🎉 no goals
                      -/


/-- `Identical` as a `Setoid`. -/
def identicalSetoid : Setoid PGame :=
  ⟨Identical, Identical.refl, Identical.symm, Identical.trans⟩


instance : IsRefl PGame (· ≡ ·) := ⟨Identical.refl⟩

instance : IsSymm PGame (· ≡ ·) := ⟨fun _ _ ↦ Identical.symm⟩

instance : IsTrans PGame (· ≡ ·) := ⟨fun _ _ _ ↦ Identical.trans⟩

instance : IsEquiv PGame (· ≡ ·) := { }


/-- If `x` and `y` are identical, then a left move of `x` is identical to some left move of `y`. -/
lemma Identical.moveLeft : ∀ {x y}, x ≡ y →
    ∀ i, ∃ j, x.moveLeft i ≡ y.moveLeft j
  | mk _ _ _ _, mk _ _ _ _, ⟨hl, _⟩, i => hl.1 i


/-- If `x` and `y` are identical, then a right move of `x` is identical to some right move of `y`.
-/
lemma Identical.moveRight : ∀ {x y}, x ≡ y →
    ∀ i, ∃ j, x.moveRight i ≡ y.moveRight j
  | mk _ _ _ _, mk _ _ _ _, ⟨_, hr⟩, i => hr.1 i


                                                                /-
                                                                  x y : SetTheory.PGame
                                                                  h : Eq x y
                                                                  ⊢ x.Identical y
                                                                -/
theorem identical_of_eq {x y : PGame} (h : x = y) : x ≡ y := by subst h; rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- Uses `∈ₗ` and `∈ᵣ` instead of `≡`. -/
theorem identical_iff' : ∀ {x y : PGame}, x ≡ y ↔
    ((∀ i, x.moveLeft i ∈ₗ y) ∧ (∀ j, y.moveLeft j ∈ₗ x)) ∧
      ((∀ i, x.moveRight i ∈ᵣ y) ∧ (∀ j, y.moveRight j ∈ᵣ x))
  | mk xl xr xL xR, mk yl yr yL yR => by
    /-
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      ⊢ Iff ((SetTheory.PGame.mk xl xr xL xR).Identical (SetTheory.PGame.mk yl yr yL …
    -/
    convert identical_iff <;>
    /-
      case h.e'_2.h.e'_1.a
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      ⊢ Iff (And (∀ (i : (SetTheory.PGame.mk xl xr xL xR).LeftMoves), ((SetTheory.PG …
    -/
    dsimp [Relator.BiTotal, Relator.LeftTotal, Relator.RightTotal] <;>
    /-
      case h.e'_2.h.e'_1.a
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      ⊢ Iff (And (∀ (i : xl), (xL i).memₗ (SetTheory.PGame.mk yl yr yL yR)) (∀ (j :  …
    -/
    congr! <;>
    /-
      case h.e'_2.h.e'_1.a.a.h.e'_2.h.a
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      a✝ : yl
      ⊢ Iff ((yL a✝).memₗ (SetTheory.PGame.mk xl xr xL xR)) (Exists fun a => (xL a). …
    -/
    /-
      🎉 no goals
    -/
    exact exists_congr <| fun _ ↦ identical_comm
    /-
      🎉 no goals
    -/


theorem memₗ.congr_right : ∀ {x y : PGame},
    x ≡ y → (∀ {w : PGame}, w ∈ₗ x ↔ w ∈ₗ y)
  | mk _ _ _ _, mk _ _ _ _, ⟨⟨h₁, h₂⟩, _⟩, _w =>
    ⟨fun ⟨i, hi⟩ ↦ (h₁ i).imp (fun _ ↦ hi.trans),
      fun ⟨j, hj⟩ ↦ (h₂ j).imp (fun _ hi ↦ hj.trans hi.symm)⟩


theorem memᵣ.congr_right : ∀ {x y : PGame},
    x ≡ y → (∀ {w : PGame}, w ∈ᵣ x ↔ w ∈ᵣ y)
  | mk _ _ _ _, mk _ _ _ _, ⟨_, ⟨h₁, h₂⟩⟩, _w =>
    ⟨fun ⟨i, hi⟩ ↦ (h₁ i).imp (fun _ ↦ hi.trans),
      fun ⟨j, hj⟩ ↦ (h₂ j).imp (fun _ hi ↦ hj.trans hi.symm)⟩


theorem memₗ.congr_left : ∀ {x y : PGame},
    x ≡ y → (∀ {w : PGame}, x ∈ₗ w ↔ y ∈ₗ w)
  | _, _, h, mk _ _ _ _ => ⟨fun ⟨i, hi⟩ ↦ ⟨i, h.symm.trans hi⟩, fun ⟨i, hi⟩ ↦ ⟨i, h.trans hi⟩⟩


theorem memᵣ.congr_left : ∀ {x y : PGame},
    x ≡ y → (∀ {w : PGame}, x ∈ᵣ w ↔ y ∈ᵣ w)
  | _, _, h, mk _ _ _ _ => ⟨fun ⟨i, hi⟩ ↦ ⟨i, h.symm.trans hi⟩, fun ⟨i, hi⟩ ↦ ⟨i, h.trans hi⟩⟩


lemma Identical.ext : ∀ {x y}, (∀ z, z ∈ₗ x ↔ z ∈ₗ y) → (∀ z, z ∈ᵣ x ↔ z ∈ᵣ y) → x ≡ y
  | mk _ _ _ _, mk _ _ _ _, hl, hr => identical_iff'.mpr
    ⟨⟨fun i ↦ (hl _).mp ⟨i, refl _⟩, fun j ↦ (hl _).mpr ⟨j, refl _⟩⟩,
      ⟨fun i ↦ (hr _).mp ⟨i, refl _⟩, fun j ↦ (hr _).mpr ⟨j, refl _⟩⟩⟩


lemma Identical.ext_iff {x y} : x ≡ y ↔ (∀ z, z ∈ₗ x ↔ z ∈ₗ y) ∧ (∀ z, z ∈ᵣ x ↔ z ∈ᵣ y) :=
  ⟨fun h ↦ ⟨@memₗ.congr_right _ _ h, @memᵣ.congr_right _ _ h⟩, fun h ↦ h.elim Identical.ext⟩


lemma Identical.congr_right {x y z} (h : x ≡ y) : z ≡ x ↔ z ≡ y :=
  ⟨fun hz ↦ hz.trans h, fun hz ↦ hz.trans h.symm⟩


lemma Identical.congr_left {x y z} (h : x ≡ y) : x ≡ z ↔ y ≡ z :=
  ⟨fun hz ↦ h.symm.trans hz, fun hz ↦ h.trans hz⟩


/-- Show `x ≡ y` by giving an explicit correspondence between the moves of `x` and `y`. -/
lemma Identical.of_fn {x y : PGame}
    (l : x.LeftMoves → y.LeftMoves) (il : y.LeftMoves → x.LeftMoves)
    (r : x.RightMoves → y.RightMoves) (ir : y.RightMoves → x.RightMoves)
    (hl : ∀ i, x.moveLeft i ≡ y.moveLeft (l i))
    (hil : ∀ i, x.moveLeft (il i) ≡ y.moveLeft i)
    (hr : ∀ i, x.moveRight i ≡ y.moveRight (r i))
    (hir : ∀ i, x.moveRight (ir i) ≡ y.moveRight i) : x ≡ y :=
  identical_iff.mpr
    ⟨⟨fun i ↦ ⟨l i, hl i⟩, fun i ↦ ⟨il i, hil i⟩⟩, ⟨fun i ↦ ⟨r i, hr i⟩, fun i ↦ ⟨ir i, hir i⟩⟩⟩


lemma Identical.of_equiv {x y : PGame}
    (l : x.LeftMoves ≃ y.LeftMoves) (r : x.RightMoves ≃ y.RightMoves)
    (hl : ∀ i, x.moveLeft i ≡ y.moveLeft (l i)) (hr : ∀ i, x.moveRight i ≡ y.moveRight (r i)) :
    x ≡ y :=
  .of_fn l l.symm r r.symm hl (by simpa using hl <| l.symm ·) hr (by simpa using hr <| r.symm ·)


/-- The less or equal relation on pre-games.

If `0 ≤ x`, then Left can win `x` as the second player. `x ≤ y` means that `0 ≤ y - x`.
See `PGame.le_iff_sub_nonneg`. -/
instance le : LE PGame :=
  ⟨Sym2.GameAdd.fix wf_isOption fun x y le =>
      (∀ i, ¬le y (x.moveLeft i) (Sym2.GameAdd.snd_fst <| IsOption.moveLeft i)) ∧
        ∀ j, ¬le (y.moveRight j) x (Sym2.GameAdd.fst_snd <| IsOption.moveRight j)⟩


/-- The less or fuzzy relation on pre-games. `x ⧏ y` is defined as `¬ y ≤ x`.

If `0 ⧏ x`, then Left can win `x` as the first player. `x ⧏ y` means that `0 ⧏ y - x`.
See `PGame.lf_iff_sub_zero_lf`. -/
def LF (x y : PGame) : Prop :=
  ¬y ≤ x


@[inherit_doc]
scoped infixl:50 " ⧏ " => PGame.LF


@[simp]
protected theorem not_le {x y : PGame} : ¬x ≤ y ↔ y ⧏ x :=
  Iff.rfl


@[simp]
theorem not_lf {x y : PGame} : ¬x ⧏ y ↔ y ≤ x :=
  Classical.not_not


theorem _root_.LE.le.not_gf {x y : PGame} : x ≤ y → ¬y ⧏ x :=
  not_lf.2


theorem LF.not_ge {x y : PGame} : x ⧏ y → ¬y ≤ x :=
  id


/-- Definition of `x ≤ y` on pre-games, in terms of `⧏`.

The ordering here is chosen so that `And.left` refer to moves by Left, and `And.right` refer to
moves by Right. -/
theorem le_iff_forall_lf {x y : PGame} :
    x ≤ y ↔ (∀ i, x.moveLeft i ⧏ y) ∧ ∀ j, x ⧏ y.moveRight j := by
  /-
    x y : SetTheory.PGame
    ⊢ Iff (LE.le x y) (And (∀ (i : x.LeftMoves), (x.moveLeft i).LF y) (∀ (j : y.Ri …
  -/
  unfold LE.le le
  /-
    x y : SetTheory.PGame
    ⊢ Iff ({ le := Sym2.GameAdd.fix SetTheory.PGame.wf_isOption fun x y le => And  …
  -/
  simp only
  /-
    x y : SetTheory.PGame
    ⊢ Iff (Sym2.GameAdd.fix SetTheory.PGame.wf_isOption (fun x y le => And (∀ (i : …
  -/
  rw [Sym2.GameAdd.fix_eq]
  /-
    x y : SetTheory.PGame
    ⊢ Iff (And (∀ (i : x.LeftMoves), Not ((fun a' b' x => Sym2.GameAdd.fix SetTheo …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Definition of `x ≤ y` on pre-games built using the constructor. -/
@[simp]
theorem mk_le_mk {xl xr xL xR yl yr yL yR} :
    mk xl xr xL xR ≤ mk yl yr yL yR ↔ (∀ i, xL i ⧏ mk yl yr yL yR) ∧ ∀ j, mk xl xr xL xR ⧏ yR j :=
  le_iff_forall_lf


theorem le_of_forall_lf {x y : PGame} (h₁ : ∀ i, x.moveLeft i ⧏ y) (h₂ : ∀ j, x ⧏ y.moveRight j) :
    x ≤ y :=
  le_iff_forall_lf.2 ⟨h₁, h₂⟩


/-- Definition of `x ⧏ y` on pre-games, in terms of `≤`.

The ordering here is chosen so that `or.inl` refer to moves by Left, and `or.inr` refer to
moves by Right. -/
theorem lf_iff_exists_le {x y : PGame} :
    x ⧏ y ↔ (∃ i, x ≤ y.moveLeft i) ∨ ∃ j, x.moveRight j ≤ y := by
  /-
    x y : SetTheory.PGame
    ⊢ Iff (x.LF y) (Or (Exists fun i => LE.le x (y.moveLeft i)) (Exists fun j => L …
  -/
  rw [LF, le_iff_forall_lf, not_and_or]
  /-
    x y : SetTheory.PGame
    ⊢ Iff (Or (Not (∀ (i : y.LeftMoves), (y.moveLeft i).LF x)) (Not (∀ (j : x.Righ …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Definition of `x ⧏ y` on pre-games built using the constructor. -/
@[simp]
theorem mk_lf_mk {xl xr xL xR yl yr yL yR} :
    mk xl xr xL xR ⧏ mk yl yr yL yR ↔ (∃ i, mk xl xr xL xR ≤ yL i) ∨ ∃ j, xR j ≤ mk yl yr yL yR :=
  lf_iff_exists_le


theorem le_or_gf (x y : PGame) : x ≤ y ∨ y ⧏ x := by
  /-
    x y : SetTheory.PGame
    ⊢ Or (LE.le x y) (y.LF x)
  -/
  rw [← PGame.not_le]
  /-
    x y : SetTheory.PGame
    ⊢ Or (LE.le x y) (Not (LE.le x y))
  -/
  apply em
  /-
    🎉 no goals
  -/


theorem moveLeft_lf_of_le {x y : PGame} (h : x ≤ y) (i) : x.moveLeft i ⧏ y :=
  (le_iff_forall_lf.1 h).1 i


alias _root_.LE.le.moveLeft_lf := moveLeft_lf_of_le


theorem lf_moveRight_of_le {x y : PGame} (h : x ≤ y) (j) : x ⧏ y.moveRight j :=
  (le_iff_forall_lf.1 h).2 j


alias _root_.LE.le.lf_moveRight := lf_moveRight_of_le


theorem lf_of_moveRight_le {x y : PGame} {j} (h : x.moveRight j ≤ y) : x ⧏ y :=
  lf_iff_exists_le.2 <| Or.inr ⟨j, h⟩


theorem lf_of_le_moveLeft {x y : PGame} {i} (h : x ≤ y.moveLeft i) : x ⧏ y :=
  lf_iff_exists_le.2 <| Or.inl ⟨i, h⟩


theorem lf_of_le_mk {xl xr xL xR y} : mk xl xr xL xR ≤ y → ∀ i, xL i ⧏ y :=
  moveLeft_lf_of_le


theorem lf_of_mk_le {x yl yr yL yR} : x ≤ mk yl yr yL yR → ∀ j, x ⧏ yR j :=
  lf_moveRight_of_le


theorem mk_lf_of_le {xl xr y j} (xL) {xR : xr → PGame} : xR j ≤ y → mk xl xr xL xR ⧏ y :=
  @lf_of_moveRight_le (mk _ _ _ _) y j


theorem lf_mk_of_le {x yl yr} {yL : yl → PGame} (yR) {i} : x ≤ yL i → x ⧏ mk yl yr yL yR :=
  @lf_of_le_moveLeft x (mk _ _ _ _) i

/- We prove that `x ≤ y → y ≤ z → x ≤ z` inductively, by also simultaneously proving its cyclic
reorderings. This auxiliary lemma is used during said induction. -/

private theorem le_trans_aux {x y z : PGame}
    (h₁ : ∀ {i}, y ≤ z → z ≤ x.moveLeft i → y ≤ x.moveLeft i)
    (h₂ : ∀ {j}, z.moveRight j ≤ x → x ≤ y → z.moveRight j ≤ y) (hxy : x ≤ y) (hyz : y ≤ z) :
    x ≤ z :=
  le_of_forall_lf (fun i => PGame.not_le.1 fun h => (h₁ hyz h).not_gf <| hxy.moveLeft_lf i)
    fun j => PGame.not_le.1 fun h => (h₂ h hxy).not_gf <| hyz.lf_moveRight j


instance : Preorder PGame :=
  { PGame.le with
    le_refl := fun x => by
      /-
        xl xr : Type u
        x : SetTheory.PGame
        ⊢ LE.le x x
      -/
      induction x with | mk _ _ _ _ IHl IHr => _
      exact
        le_of_forall_lf (fun i => lf_of_le_moveLeft (IHl i)) fun i => lf_of_moveRight_le (IHr i)
    le_trans := by
      suffices
        ∀ {x y z : PGame},
          (x ≤ y → y ≤ z → x ≤ z) ∧ (y ≤ z → z ≤ x → y ≤ x) ∧ (z ≤ x → x ≤ y → z ≤ y) from
        fun x y z => this.1
      /-
        xl xr : Type u
        ⊢ ∀ {x y z : SetTheory.PGame}, And (LE.le x y → LE.le y z → LE.le x z) (And (L …
      -/
      intro x y z
      /-
        xl xr : Type u
        x y z : SetTheory.PGame
        ⊢ And (LE.le x y → LE.le y z → LE.le x z) (And (LE.le y z → LE.le z x → LE.le  …
      -/
      induction' x with xl xr xL xR IHxl IHxr generalizing y z
      /-
        case mk
        xl✝ xr✝ : Type u
        xl xr : Type ?u.39614
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        IHxl : ∀ (a : xl) {y z : SetTheory.PGame}, And (LE.le (xL a) y → LE.le y z → L …
        IHxr : ∀ (a : xr) {y z : SetTheory.PGame}, And (LE.le (xR a) y → LE.le y z → L …
        y z : SetTheory.PGame
        ⊢ And (LE.le (SetTheory.PGame.mk xl xr xL xR) y → LE.le y z → LE.le (SetTheory …
      -/
      induction' y with yl yr yL yR IHyl IHyr generalizing z
      /-
        case mk.mk
        xl✝ xr✝ : Type u
        xl xr : Type ?u.39614
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        IHxl : ∀ (a : xl) {y z : SetTheory.PGame}, And (LE.le (xL a) y → LE.le y z → L …
        IHxr : ∀ (a : xr) {y z : SetTheory.PGame}, And (LE.le (xR a) y → LE.le y z → L …
        yl yr : Type ?u.39614
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        IHyl : ∀ (a : yl) {z : SetTheory.PGame}, And (LE.le (SetTheory.PGame.mk xl xr  …
        IHyr : ∀ (a : yr) {z : SetTheory.PGame}, And (LE.le (SetTheory.PGame.mk xl xr  …
        z : SetTheory.PGame
        ⊢ And (LE.le (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk yl yr yL yR) …
      -/
      induction' z with zl zr zL zR IHzl IHzr
      exact
        ⟨le_trans_aux (fun {i} => (IHxl i).2.1) fun {j} => (IHzr j).2.2,
          le_trans_aux (fun {i} => (IHyl i).2.2) fun {j} => (IHxr j).1,
          le_trans_aux (fun {i} => (IHzl i).1) fun {j} => (IHyr j).2.1⟩
    lt := fun x y => x ≤ y ∧ x ⧏ y }


lemma Identical.le : ∀ {x y}, x ≡ y → x ≤ y
  | mk _ _ _ _, mk _ _ _ _, ⟨hL, hR⟩ => le_of_forall_lf
    (fun i ↦ let ⟨_, hj⟩ := hL.1 i; lf_of_le_moveLeft hj.le)
    (fun i ↦ let ⟨_, hj⟩ := hR.2 i; lf_of_moveRight_le hj.le)


lemma Identical.ge {x y} (h : x ≡ y) : y ≤ x := h.symm.le


theorem lt_iff_le_and_lf {x y : PGame} : x < y ↔ x ≤ y ∧ x ⧏ y :=
  Iff.rfl


theorem lt_of_le_of_lf {x y : PGame} (h₁ : x ≤ y) (h₂ : x ⧏ y) : x < y :=
  ⟨h₁, h₂⟩


theorem lf_of_lt {x y : PGame} (h : x < y) : x ⧏ y :=
  h.2


alias _root_.LT.lt.lf := lf_of_lt


theorem lf_irrefl (x : PGame) : ¬x ⧏ x :=
  le_rfl.not_gf


instance : IsIrrefl _ (· ⧏ ·) :=
  ⟨lf_irrefl⟩


@[trans]
theorem lf_of_le_of_lf {x y z : PGame} (h₁ : x ≤ y) (h₂ : y ⧏ z) : x ⧏ z := by
  /-
    x y z : SetTheory.PGame
    h₁ : LE.le x y
    h₂ : y.LF z
    ⊢ x.LF z
  -/
  rw [← PGame.not_le] at h₂ ⊢
  /-
    x y z : SetTheory.PGame
    h₁ : LE.le x y
    h₂ : Not (LE.le z y)
    ⊢ Not (LE.le z x)
  -/
  exact fun h₃ => h₂ (h₃.trans h₁)
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): added instance

instance : Trans (· ≤ ·) (· ⧏ ·) (· ⧏ ·) := ⟨lf_of_le_of_lf⟩


@[trans]
theorem lf_of_lf_of_le {x y z : PGame} (h₁ : x ⧏ y) (h₂ : y ≤ z) : x ⧏ z := by
  /-
    x y z : SetTheory.PGame
    h₁ : x.LF y
    h₂ : LE.le y z
    ⊢ x.LF z
  -/
  rw [← PGame.not_le] at h₁ ⊢
  /-
    x y z : SetTheory.PGame
    h₁ : Not (LE.le y x)
    h₂ : LE.le y z
    ⊢ Not (LE.le z x)
  -/
  exact fun h₃ => h₁ (h₂.trans h₃)
  /-
    🎉 no goals
  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): added instance

instance : Trans (· ⧏ ·) (· ≤ ·) (· ⧏ ·) := ⟨lf_of_lf_of_le⟩


alias _root_.LE.le.trans_lf := lf_of_le_of_lf


alias LF.trans_le := lf_of_lf_of_le


@[trans]
theorem lf_of_lt_of_lf {x y z : PGame} (h₁ : x < y) (h₂ : y ⧏ z) : x ⧏ z :=
  h₁.le.trans_lf h₂


@[trans]
theorem lf_of_lf_of_lt {x y z : PGame} (h₁ : x ⧏ y) (h₂ : y < z) : x ⧏ z :=
  h₁.trans_le h₂.le


alias _root_.LT.lt.trans_lf := lf_of_lt_of_lf


alias LF.trans_lt := lf_of_lf_of_lt


theorem moveLeft_lf {x : PGame} : ∀ i, x.moveLeft i ⧏ x :=
  le_rfl.moveLeft_lf


theorem lf_moveRight {x : PGame} : ∀ j, x ⧏ x.moveRight j :=
  le_rfl.lf_moveRight


theorem lf_mk {xl xr} (xL : xl → PGame) (xR : xr → PGame) (i) : xL i ⧏ mk xl xr xL xR :=
  @moveLeft_lf (mk _ _ _ _) i


theorem mk_lf {xl xr} (xL : xl → PGame) (xR : xr → PGame) (j) : mk xl xr xL xR ⧏ xR j :=
  @lf_moveRight (mk _ _ _ _) j


/-- This special case of `PGame.le_of_forall_lf` is useful when dealing with surreals, where `<` is
preferred over `⧏`. -/
theorem le_of_forall_lt {x y : PGame} (h₁ : ∀ i, x.moveLeft i < y) (h₂ : ∀ j, x < y.moveRight j) :
    x ≤ y :=
  le_of_forall_lf (fun i => (h₁ i).lf) fun i => (h₂ i).lf


/-- The definition of `x ≤ y` on pre-games, in terms of `≤` two moves later.

Note that it's often more convenient to use `le_iff_forall_lf`, which only unfolds the definition by
one step. -/
theorem le_def {x y : PGame} :
    x ≤ y ↔
      (∀ i, (∃ i', x.moveLeft i ≤ y.moveLeft i') ∨ ∃ j, (x.moveLeft i).moveRight j ≤ y) ∧
        ∀ j, (∃ i, x ≤ (y.moveRight j).moveLeft i) ∨ ∃ j', x.moveRight j' ≤ y.moveRight j := by
  /-
    x y : SetTheory.PGame
    ⊢ Iff (LE.le x y) (And (∀ (i : x.LeftMoves), Or (Exists fun i' => LE.le (x.mov …
  -/
  rw [le_iff_forall_lf]
  conv =>
    lhs
    simp only [lf_iff_exists_le]


/-- The definition of `x ⧏ y` on pre-games, in terms of `⧏` two moves later.

Note that it's often more convenient to use `lf_iff_exists_le`, which only unfolds the definition by
one step. -/
theorem lf_def {x y : PGame} :
    x ⧏ y ↔
      (∃ i, (∀ i', x.moveLeft i' ⧏ y.moveLeft i) ∧ ∀ j, x ⧏ (y.moveLeft i).moveRight j) ∨
        ∃ j, (∀ i, (x.moveRight j).moveLeft i ⧏ y) ∧ ∀ j', x.moveRight j ⧏ y.moveRight j' := by
  /-
    x y : SetTheory.PGame
    ⊢ Iff (x.LF y) (Or (Exists fun i => And (∀ (i' : x.LeftMoves), (x.moveLeft i') …
  -/
  rw [lf_iff_exists_le]
  conv =>
    lhs
    simp only [le_iff_forall_lf]


/-- The definition of `0 ≤ x` on pre-games, in terms of `0 ⧏`. -/
theorem zero_le_lf {x : PGame} : 0 ≤ x ↔ ∀ j, 0 ⧏ x.moveRight j := by
  /-
    x : SetTheory.PGame
    ⊢ Iff (LE.le 0 x) (∀ (j : x.RightMoves), SetTheory.PGame.LF 0 (x.moveRight j))
  -/
  rw [le_iff_forall_lf]
  /-
    x : SetTheory.PGame
    ⊢ Iff (And (∀ (i : SetTheory.PGame.LeftMoves 0), (SetTheory.PGame.moveLeft 0 i …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The definition of `x ≤ 0` on pre-games, in terms of `⧏ 0`. -/
theorem le_zero_lf {x : PGame} : x ≤ 0 ↔ ∀ i, x.moveLeft i ⧏ 0 := by
  /-
    x : SetTheory.PGame
    ⊢ Iff (LE.le x 0) (∀ (i : x.LeftMoves), (x.moveLeft i).LF 0)
  -/
  rw [le_iff_forall_lf]
  /-
    x : SetTheory.PGame
    ⊢ Iff (And (∀ (i : x.LeftMoves), (x.moveLeft i).LF 0) (∀ (j : SetTheory.PGame. …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The definition of `0 ⧏ x` on pre-games, in terms of `0 ≤`. -/
theorem zero_lf_le {x : PGame} : 0 ⧏ x ↔ ∃ i, 0 ≤ x.moveLeft i := by
  /-
    x : SetTheory.PGame
    ⊢ Iff (SetTheory.PGame.LF 0 x) (Exists fun i => LE.le 0 (x.moveLeft i))
  -/
  rw [lf_iff_exists_le]
  /-
    x : SetTheory.PGame
    ⊢ Iff (Or (Exists fun i => LE.le 0 (x.moveLeft i)) (Exists fun j => LE.le (Set …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The definition of `x ⧏ 0` on pre-games, in terms of `≤ 0`. -/
theorem lf_zero_le {x : PGame} : x ⧏ 0 ↔ ∃ j, x.moveRight j ≤ 0 := by
  /-
    x : SetTheory.PGame
    ⊢ Iff (x.LF 0) (Exists fun j => LE.le (x.moveRight j) 0)
  -/
  rw [lf_iff_exists_le]
  /-
    x : SetTheory.PGame
    ⊢ Iff (Or (Exists fun i => LE.le x (SetTheory.PGame.moveLeft 0 i)) (Exists fun …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The definition of `0 ≤ x` on pre-games, in terms of `0 ≤` two moves later. -/
theorem zero_le {x : PGame} : 0 ≤ x ↔ ∀ j, ∃ i, 0 ≤ (x.moveRight j).moveLeft i := by
  /-
    x : SetTheory.PGame
    ⊢ Iff (LE.le 0 x) (∀ (j : x.RightMoves), Exists fun i => LE.le 0 ((x.moveRight …
  -/
  rw [le_def]
  /-
    x : SetTheory.PGame
    ⊢ Iff (And (∀ (i : SetTheory.PGame.LeftMoves 0), Or (Exists fun i' => LE.le (S …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The definition of `x ≤ 0` on pre-games, in terms of `≤ 0` two moves later. -/
theorem le_zero {x : PGame} : x ≤ 0 ↔ ∀ i, ∃ j, (x.moveLeft i).moveRight j ≤ 0 := by
  /-
    x : SetTheory.PGame
    ⊢ Iff (LE.le x 0) (∀ (i : x.LeftMoves), Exists fun j => LE.le ((x.moveLeft i). …
  -/
  rw [le_def]
  /-
    x : SetTheory.PGame
    ⊢ Iff (And (∀ (i : x.LeftMoves), Or (Exists fun i' => LE.le (x.moveLeft i) (Se …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The definition of `0 ⧏ x` on pre-games, in terms of `0 ⧏` two moves later. -/
theorem zero_lf {x : PGame} : 0 ⧏ x ↔ ∃ i, ∀ j, 0 ⧏ (x.moveLeft i).moveRight j := by
  /-
    x : SetTheory.PGame
    ⊢ Iff (SetTheory.PGame.LF 0 x) (Exists fun i => ∀ (j : (x.moveLeft i).RightMov …
  -/
  rw [lf_def]
  /-
    x : SetTheory.PGame
    ⊢ Iff (Or (Exists fun i => And (∀ (i' : SetTheory.PGame.LeftMoves 0), (SetTheo …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The definition of `x ⧏ 0` on pre-games, in terms of `⧏ 0` two moves later. -/
theorem lf_zero {x : PGame} : x ⧏ 0 ↔ ∃ j, ∀ i, (x.moveRight j).moveLeft i ⧏ 0 := by
  /-
    x : SetTheory.PGame
    ⊢ Iff (x.LF 0) (Exists fun j => ∀ (i : (x.moveRight j).LeftMoves), ((x.moveRig …
  -/
  rw [lf_def]
  /-
    x : SetTheory.PGame
    ⊢ Iff (Or (Exists fun i => And (∀ (i' : x.LeftMoves), (x.moveLeft i').LF (SetT …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_le_of_isEmpty_rightMoves (x : PGame) [IsEmpty x.RightMoves] : 0 ≤ x :=
  zero_le.2 isEmptyElim


@[simp]
theorem le_zero_of_isEmpty_leftMoves (x : PGame) [IsEmpty x.LeftMoves] : x ≤ 0 :=
  le_zero.2 isEmptyElim


/-- Given a game won by the right player when they play second, provide a response to any move by
left. -/
noncomputable def rightResponse {x : PGame} (h : x ≤ 0) (i : x.LeftMoves) :
    (x.moveLeft i).RightMoves :=
  Classical.choose <| (le_zero.1 h) i


/-- Show that the response for right provided by `rightResponse` preserves the right-player-wins
condition. -/
theorem rightResponse_spec {x : PGame} (h : x ≤ 0) (i : x.LeftMoves) :
    (x.moveLeft i).moveRight (rightResponse h i) ≤ 0 :=
  Classical.choose_spec <| (le_zero.1 h) i


/-- Given a game won by the left player when they play second, provide a response to any move by
right. -/
noncomputable def leftResponse {x : PGame} (h : 0 ≤ x) (j : x.RightMoves) :
    (x.moveRight j).LeftMoves :=
  Classical.choose <| (zero_le.1 h) j


/-- Show that the response for left provided by `leftResponse` preserves the left-player-wins
condition. -/
theorem leftResponse_spec {x : PGame} (h : 0 ≤ x) (j : x.RightMoves) :
    0 ≤ (x.moveRight j).moveLeft (leftResponse h j) :=
  Classical.choose_spec <| (zero_le.1 h) j


/-- A small family of pre-games is bounded above. -/
lemma bddAbove_range_of_small {ι : Type*} [Small.{u} ι] (f : ι → PGame.{u}) :
    BddAbove (Set.range f) := by
  let x : PGame.{u} := ⟨Σ i, (f <| (equivShrink.{u} ι).symm i).LeftMoves, PEmpty,
    fun x ↦ moveLeft _ x.2, PEmpty.elim⟩
  /-
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → SetTheory.PGame
    x : SetTheory.PGame := SetTheory.PGame.mk (Sigma fun i => (f ((equivShrink ι). …
    ⊢ BddAbove (Set.range f)
  -/
  refine ⟨x, Set.forall_mem_range.2 fun i ↦ ?_⟩
  /-
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → SetTheory.PGame
    x : SetTheory.PGame := SetTheory.PGame.mk (Sigma fun i => (f ((equivShrink ι). …
    i : ι
    ⊢ LE.le (f i) x
  -/
  rw [← (equivShrink ι).symm_apply_apply i, le_iff_forall_lf]
  /-
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → SetTheory.PGame
    x : SetTheory.PGame := SetTheory.PGame.mk (Sigma fun i => (f ((equivShrink ι). …
    i : ι
    ⊢ And (∀ (i_1 : (f ((equivShrink ι).symm ((equivShrink ι) i))).LeftMoves), ((f …
  -/
  simpa [x] using fun j ↦ @moveLeft_lf x ⟨equivShrink ι i, j⟩
  /-
    🎉 no goals
  -/


/-- A small set of pre-games is bounded above. -/
lemma bddAbove_of_small (s : Set PGame.{u}) [Small.{u} s] : BddAbove s := by
  /-
    s : Set SetTheory.PGame
    inst✝ : Small.{u, u + 1} ↑s
    ⊢ BddAbove s
  -/
  simpa using bddAbove_range_of_small (Subtype.val : s → PGame.{u})
  /-
    🎉 no goals
  -/


/-- A small family of pre-games is bounded below. -/
lemma bddBelow_range_of_small {ι : Type*} [Small.{u} ι] (f : ι → PGame.{u}) :
    BddBelow (Set.range f) := by
  let x : PGame.{u} := ⟨PEmpty, Σ i, (f <| (equivShrink.{u} ι).symm i).RightMoves, PEmpty.elim,
    fun x ↦ moveRight _ x.2⟩
  /-
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → SetTheory.PGame
    x : SetTheory.PGame := SetTheory.PGame.mk PEmpty.{u + 1} (Sigma fun i => (f (( …
    ⊢ BddBelow (Set.range f)
  -/
  refine ⟨x, Set.forall_mem_range.2 fun i ↦ ?_⟩
  /-
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → SetTheory.PGame
    x : SetTheory.PGame := SetTheory.PGame.mk PEmpty.{u + 1} (Sigma fun i => (f (( …
    i : ι
    ⊢ LE.le x (f i)
  -/
  rw [← (equivShrink ι).symm_apply_apply i, le_iff_forall_lf]
  /-
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → SetTheory.PGame
    x : SetTheory.PGame := SetTheory.PGame.mk PEmpty.{u + 1} (Sigma fun i => (f (( …
    i : ι
    ⊢ And (∀ (i_1 : x.LeftMoves), (x.moveLeft i_1).LF (f ((equivShrink ι).symm ((e …
  -/
  simpa [x] using fun j ↦ @lf_moveRight x ⟨equivShrink ι i, j⟩
  /-
    🎉 no goals
  -/


/-- A small set of pre-games is bounded below. -/
lemma bddBelow_of_small (s : Set PGame.{u}) [Small.{u} s] : BddBelow s := by
  /-
    s : Set SetTheory.PGame
    inst✝ : Small.{u, u + 1} ↑s
    ⊢ BddBelow s
  -/
  simpa using bddBelow_range_of_small (Subtype.val : s → PGame.{u})
  /-
    🎉 no goals
  -/


/-- The equivalence relation on pre-games. Two pre-games `x`, `y` are equivalent if `x ≤ y` and
`y ≤ x`.

If `x ≈ 0`, then the second player can always win `x`. -/
def Equiv (x y : PGame) : Prop :=
  x ≤ y ∧ y ≤ x

-- Porting note: deleted the scoped notation due to notation overloading with the setoid
-- instance and this causes the PGame.equiv docstring to not show up on hover.


instance : IsEquiv _ PGame.Equiv where
  refl _ := ⟨le_rfl, le_rfl⟩
  trans := fun _ _ _ ⟨xy, yx⟩ ⟨yz, zy⟩ => ⟨xy.trans yz, zy.trans yx⟩
  symm _ _ := And.symm

-- Porting note: moved the setoid instance from Basic.lean to here


instance setoid : Setoid PGame :=
  ⟨Equiv, refl, symm, Trans.trans⟩


theorem equiv_def {x y : PGame} : x ≈ y ↔ x ≤ y ∧ y ≤ x := Iff.rfl


theorem Equiv.le {x y : PGame} (h : x ≈ y) : x ≤ y :=
  h.1


theorem Equiv.ge {x y : PGame} (h : x ≈ y) : y ≤ x :=
  h.2


@[refl, simp]
theorem equiv_rfl {x : PGame} : x ≈ x :=
  refl x


theorem equiv_refl (x : PGame) : x ≈ x :=
  refl x


@[symm]
protected theorem Equiv.symm {x y : PGame} : (x ≈ y) → (y ≈ x) :=
  symm


@[trans]
protected theorem Equiv.trans {x y z : PGame} : (x ≈ y) → (y ≈ z) → (x ≈ z) :=
  _root_.trans


protected theorem equiv_comm {x y : PGame} : (x ≈ y) ↔ (y ≈ x) :=
  comm


                                                            /-
                                                              x y : SetTheory.PGame
                                                              h : Eq x y
                                                              ⊢ HasEquiv.Equiv x y
                                                            -/
theorem equiv_of_eq {x y : PGame} (h : x = y) : x ≈ y := by subst h; rfl
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


lemma Identical.equiv {x y} (h : x ≡ y) : x ≈ y := ⟨h.le, h.ge⟩


@[trans]
theorem le_of_le_of_equiv {x y z : PGame} (h₁ : x ≤ y) (h₂ : y ≈ z) : x ≤ z :=
  h₁.trans h₂.1


instance : Trans
    ((· ≤ ·) : PGame → PGame → Prop)
    ((· ≈ ·) : PGame → PGame → Prop)
    ((· ≤ ·) : PGame → PGame → Prop) where
  trans := le_of_le_of_equiv


@[trans]
theorem le_of_equiv_of_le {x y z : PGame} (h₁ : x ≈ y) : y ≤ z → x ≤ z :=
  h₁.1.trans


instance : Trans
    ((· ≈ ·) : PGame → PGame → Prop)
    ((· ≤ ·) : PGame → PGame → Prop)
    ((· ≤ ·) : PGame → PGame → Prop) where
  trans := le_of_equiv_of_le


theorem LF.not_equiv {x y : PGame} (h : x ⧏ y) : ¬(x ≈ y) := fun h' => h.not_ge h'.2


theorem LF.not_equiv' {x y : PGame} (h : x ⧏ y) : ¬(y ≈ x) := fun h' => h.not_ge h'.1


theorem LF.not_gt {x y : PGame} (h : x ⧏ y) : ¬y < x := fun h' => h.not_ge h'.le


theorem le_congr_imp {x₁ y₁ x₂ y₂ : PGame} (hx : x₁ ≈ x₂) (hy : y₁ ≈ y₂) (h : x₁ ≤ y₁) : x₂ ≤ y₂ :=
  hx.2.trans (h.trans hy.1)


theorem le_congr {x₁ y₁ x₂ y₂ : PGame} (hx : x₁ ≈ x₂) (hy : y₁ ≈ y₂) : x₁ ≤ y₁ ↔ x₂ ≤ y₂ :=
  ⟨le_congr_imp hx hy, le_congr_imp (Equiv.symm hx) (Equiv.symm hy)⟩


theorem le_congr_left {x₁ x₂ y : PGame} (hx : x₁ ≈ x₂) : x₁ ≤ y ↔ x₂ ≤ y :=
  le_congr hx equiv_rfl


theorem le_congr_right {x y₁ y₂ : PGame} (hy : y₁ ≈ y₂) : x ≤ y₁ ↔ x ≤ y₂ :=
  le_congr equiv_rfl hy


theorem lf_congr {x₁ y₁ x₂ y₂ : PGame} (hx : x₁ ≈ x₂) (hy : y₁ ≈ y₂) : x₁ ⧏ y₁ ↔ x₂ ⧏ y₂ :=
  PGame.not_le.symm.trans <| (not_congr (le_congr hy hx)).trans PGame.not_le


theorem lf_congr_imp {x₁ y₁ x₂ y₂ : PGame} (hx : x₁ ≈ x₂) (hy : y₁ ≈ y₂) : x₁ ⧏ y₁ → x₂ ⧏ y₂ :=
  (lf_congr hx hy).1


theorem lf_congr_left {x₁ x₂ y : PGame} (hx : x₁ ≈ x₂) : x₁ ⧏ y ↔ x₂ ⧏ y :=
  lf_congr hx equiv_rfl


theorem lf_congr_right {x y₁ y₂ : PGame} (hy : y₁ ≈ y₂) : x ⧏ y₁ ↔ x ⧏ y₂ :=
  lf_congr equiv_rfl hy


@[trans]
theorem lf_of_lf_of_equiv {x y z : PGame} (h₁ : x ⧏ y) (h₂ : y ≈ z) : x ⧏ z :=
  lf_congr_imp equiv_rfl h₂ h₁


instance : Trans (· ⧏ ·) (· ≈ ·) (· ⧏ ·) := ⟨lf_of_lf_of_equiv⟩


@[trans]
theorem lf_of_equiv_of_lf {x y z : PGame} (h₁ : x ≈ y) : y ⧏ z → x ⧏ z :=
  lf_congr_imp (Equiv.symm h₁) equiv_rfl


instance : Trans (· ≈ ·) (· ⧏ ·) (· ⧏ ·) := ⟨lf_of_equiv_of_lf⟩


@[trans]
theorem lt_of_lt_of_equiv {x y z : PGame} (h₁ : x < y) (h₂ : y ≈ z) : x < z :=
  h₁.trans_le h₂.1


instance : Trans
    ((· < ·) : PGame → PGame → Prop)
    ((· ≈ ·) : PGame → PGame → Prop)
    ((· < ·) : PGame → PGame → Prop) where
  trans := lt_of_lt_of_equiv


@[trans]
theorem lt_of_equiv_of_lt {x y z : PGame} (h₁ : x ≈ y) : y < z → x < z :=
  h₁.1.trans_lt


instance : Trans
    ((· ≈ ·) : PGame → PGame → Prop)
    ((· < ·) : PGame → PGame → Prop)
    ((· < ·) : PGame → PGame → Prop) where
  trans := lt_of_equiv_of_lt


theorem lt_congr_imp {x₁ y₁ x₂ y₂ : PGame} (hx : x₁ ≈ x₂) (hy : y₁ ≈ y₂) (h : x₁ < y₁) : x₂ < y₂ :=
  hx.2.trans_lt (h.trans_le hy.1)


theorem lt_congr {x₁ y₁ x₂ y₂ : PGame} (hx : x₁ ≈ x₂) (hy : y₁ ≈ y₂) : x₁ < y₁ ↔ x₂ < y₂ :=
  ⟨lt_congr_imp hx hy, lt_congr_imp (Equiv.symm hx) (Equiv.symm hy)⟩


theorem lt_congr_left {x₁ x₂ y : PGame} (hx : x₁ ≈ x₂) : x₁ < y ↔ x₂ < y :=
  lt_congr hx equiv_rfl


theorem lt_congr_right {x y₁ y₂ : PGame} (hy : y₁ ≈ y₂) : x < y₁ ↔ x < y₂ :=
  lt_congr equiv_rfl hy


theorem lt_or_equiv_of_le {x y : PGame} (h : x ≤ y) : x < y ∨ (x ≈ y) :=
  and_or_left.mp ⟨h, (em <| y ≤ x).symm.imp_left PGame.not_le.1⟩


theorem lf_or_equiv_or_gf (x y : PGame) : x ⧏ y ∨ (x ≈ y) ∨ y ⧏ x := by
  /-
    x y : SetTheory.PGame
    ⊢ Or (x.LF y) (Or (HasEquiv.Equiv x y) (y.LF x))
  -/
  by_cases h : x ⧏ y
    /-
      case pos
      x y : SetTheory.PGame
      h : x.LF y
      ⊢ Or (x.LF y) (Or (HasEquiv.Equiv x y) (y.LF x))
    -/
  · exact Or.inl h
    /-
      🎉 no goals
    -/
    /-
      case neg
      x y : SetTheory.PGame
      h : Not (x.LF y)
      ⊢ Or (x.LF y) (Or (HasEquiv.Equiv x y) (y.LF x))
    -/
  · right
    /-
      case neg.h
      x y : SetTheory.PGame
      h : Not (x.LF y)
      ⊢ Or (HasEquiv.Equiv x y) (y.LF x)
    -/
    cases' lt_or_equiv_of_le (PGame.not_lf.1 h) with h' h'
      /-
        case neg.h.inl
        x y : SetTheory.PGame
        h : Not (x.LF y)
        h' : LT.lt y x
        ⊢ Or (HasEquiv.Equiv x y) (y.LF x)
      -/
    · exact Or.inr h'.lf
      /-
        🎉 no goals
      -/
      /-
        case neg.h.inr
        x y : SetTheory.PGame
        h : Not (x.LF y)
        h' : HasEquiv.Equiv y x
        ⊢ Or (HasEquiv.Equiv x y) (y.LF x)
      -/
    · exact Or.inl (Equiv.symm h')
      /-
        🎉 no goals
      -/


theorem equiv_congr_left {y₁ y₂ : PGame} : (y₁ ≈ y₂) ↔ ∀ x₁, (x₁ ≈ y₁) ↔ (x₁ ≈ y₂) :=
  ⟨fun h _ => ⟨fun h' => Equiv.trans h' h, fun h' => Equiv.trans h' (Equiv.symm h)⟩,
   fun h => (h y₁).1 <| equiv_rfl⟩


theorem equiv_congr_right {x₁ x₂ : PGame} : (x₁ ≈ x₂) ↔ ∀ y₁, (x₁ ≈ y₁) ↔ (x₂ ≈ y₁) :=
  ⟨fun h _ => ⟨fun h' => Equiv.trans (Equiv.symm h) h', fun h' => Equiv.trans h h'⟩,
   fun h => (h x₂).2 <| equiv_rfl⟩


theorem Equiv.of_exists {x y : PGame}
    (hl₁ : ∀ i, ∃ j, x.moveLeft i ≈ y.moveLeft j) (hr₁ : ∀ i, ∃ j, x.moveRight i ≈ y.moveRight j)
    (hl₂ : ∀ j, ∃ i, x.moveLeft i ≈ y.moveLeft j) (hr₂ : ∀ j, ∃ i, x.moveRight i ≈ y.moveRight j) :
    x ≈ y := by
  /-
    x y : SetTheory.PGame
    hl₁ : ∀ (i : x.LeftMoves), Exists fun j => HasEquiv.Equiv (x.moveLeft i) (y.mo …
    hr₁ : ∀ (i : x.RightMoves), Exists fun j => HasEquiv.Equiv (x.moveRight i) (y. …
    hl₂ : ∀ (j : y.LeftMoves), Exists fun i => HasEquiv.Equiv (x.moveLeft i) (y.mo …
    hr₂ : ∀ (j : y.RightMoves), Exists fun i => HasEquiv.Equiv (x.moveRight i) (y. …
    ⊢ HasEquiv.Equiv x y
  -/
  constructor <;> refine le_def.2 ⟨?_, ?_⟩ <;> intro i
    /-
      case left.refine_1
      x y : SetTheory.PGame
      hl₁ : ∀ (i : x.LeftMoves), Exists fun j => HasEquiv.Equiv (x.moveLeft i) (y.mo …
      hr₁ : ∀ (i : x.RightMoves), Exists fun j => HasEquiv.Equiv (x.moveRight i) (y. …
      hl₂ : ∀ (j : y.LeftMoves), Exists fun i => HasEquiv.Equiv (x.moveLeft i) (y.mo …
      hr₂ : ∀ (j : y.RightMoves), Exists fun i => HasEquiv.Equiv (x.moveRight i) (y. …
      i : x.LeftMoves
      ⊢ Or (Exists fun i' => LE.le (x.moveLeft i) (y.moveLeft i')) (Exists fun j =>  …
    -/
  · obtain ⟨j, hj⟩ := hl₁ i
    /-
      case left.refine_1.intro
      x y : SetTheory.PGame
      hl₁ : ∀ (i : x.LeftMoves), Exists fun j => HasEquiv.Equiv (x.moveLeft i) (y.mo …
      hr₁ : ∀ (i : x.RightMoves), Exists fun j => HasEquiv.Equiv (x.moveRight i) (y. …
      hl₂ : ∀ (j : y.LeftMoves), Exists fun i => HasEquiv.Equiv (x.moveLeft i) (y.mo …
      hr₂ : ∀ (j : y.RightMoves), Exists fun i => HasEquiv.Equiv (x.moveRight i) (y. …
      i : x.LeftMoves
      j : y.LeftMoves
      hj : HasEquiv.Equiv (x.moveLeft i) (y.moveLeft j)
      ⊢ Or (Exists fun i' => LE.le (x.moveLeft i) (y.moveLeft i')) (Exists fun j =>  …
    -/
    exact Or.inl ⟨j, Equiv.le hj⟩
    /-
      🎉 no goals
    -/
    /-
      case left.refine_2
      x y : SetTheory.PGame
      hl₁ : ∀ (i : x.LeftMoves), Exists fun j => HasEquiv.Equiv (x.moveLeft i) (y.mo …
      hr₁ : ∀ (i : x.RightMoves), Exists fun j => HasEquiv.Equiv (x.moveRight i) (y. …
      hl₂ : ∀ (j : y.LeftMoves), Exists fun i => HasEquiv.Equiv (x.moveLeft i) (y.mo …
      hr₂ : ∀ (j : y.RightMoves), Exists fun i => HasEquiv.Equiv (x.moveRight i) (y. …
      i : y.RightMoves
      ⊢ Or (Exists fun i_1 => LE.le x ((y.moveRight i).moveLeft i_1)) (Exists fun j' …
    -/
  · obtain ⟨j, hj⟩ := hr₂ i
    /-
      case left.refine_2.intro
      x y : SetTheory.PGame
      hl₁ : ∀ (i : x.LeftMoves), Exists fun j => HasEquiv.Equiv (x.moveLeft i) (y.mo …
      hr₁ : ∀ (i : x.RightMoves), Exists fun j => HasEquiv.Equiv (x.moveRight i) (y. …
      hl₂ : ∀ (j : y.LeftMoves), Exists fun i => HasEquiv.Equiv (x.moveLeft i) (y.mo …
      hr₂ : ∀ (j : y.RightMoves), Exists fun i => HasEquiv.Equiv (x.moveRight i) (y. …
      i : y.RightMoves
      j : x.RightMoves
      hj : HasEquiv.Equiv (x.moveRight j) (y.moveRight i)
      ⊢ Or (Exists fun i_1 => LE.le x ((y.moveRight i).moveLeft i_1)) (Exists fun j' …
    -/
    exact Or.inr ⟨j, Equiv.le hj⟩
    /-
      🎉 no goals
    -/
    /-
      case right.refine_1
      x y : SetTheory.PGame
      hl₁ : ∀ (i : x.LeftMoves), Exists fun j => HasEquiv.Equiv (x.moveLeft i) (y.mo …
      hr₁ : ∀ (i : x.RightMoves), Exists fun j => HasEquiv.Equiv (x.moveRight i) (y. …
      hl₂ : ∀ (j : y.LeftMoves), Exists fun i => HasEquiv.Equiv (x.moveLeft i) (y.mo …
      hr₂ : ∀ (j : y.RightMoves), Exists fun i => HasEquiv.Equiv (x.moveRight i) (y. …
      i : y.LeftMoves
      ⊢ Or (Exists fun i' => LE.le (y.moveLeft i) (x.moveLeft i')) (Exists fun j =>  …
    -/
  · obtain ⟨j, hj⟩ := hl₂ i
    /-
      case right.refine_1.intro
      x y : SetTheory.PGame
      hl₁ : ∀ (i : x.LeftMoves), Exists fun j => HasEquiv.Equiv (x.moveLeft i) (y.mo …
      hr₁ : ∀ (i : x.RightMoves), Exists fun j => HasEquiv.Equiv (x.moveRight i) (y. …
      hl₂ : ∀ (j : y.LeftMoves), Exists fun i => HasEquiv.Equiv (x.moveLeft i) (y.mo …
      hr₂ : ∀ (j : y.RightMoves), Exists fun i => HasEquiv.Equiv (x.moveRight i) (y. …
      i : y.LeftMoves
      j : x.LeftMoves
      hj : HasEquiv.Equiv (x.moveLeft j) (y.moveLeft i)
      ⊢ Or (Exists fun i' => LE.le (y.moveLeft i) (x.moveLeft i')) (Exists fun j =>  …
    -/
    exact Or.inl ⟨j, Equiv.ge hj⟩
    /-
      🎉 no goals
    -/
    /-
      case right.refine_2
      x y : SetTheory.PGame
      hl₁ : ∀ (i : x.LeftMoves), Exists fun j => HasEquiv.Equiv (x.moveLeft i) (y.mo …
      hr₁ : ∀ (i : x.RightMoves), Exists fun j => HasEquiv.Equiv (x.moveRight i) (y. …
      hl₂ : ∀ (j : y.LeftMoves), Exists fun i => HasEquiv.Equiv (x.moveLeft i) (y.mo …
      hr₂ : ∀ (j : y.RightMoves), Exists fun i => HasEquiv.Equiv (x.moveRight i) (y. …
      i : x.RightMoves
      ⊢ Or (Exists fun i_1 => LE.le y ((x.moveRight i).moveLeft i_1)) (Exists fun j' …
    -/
  · obtain ⟨j, hj⟩ := hr₁ i
    /-
      case right.refine_2.intro
      x y : SetTheory.PGame
      hl₁ : ∀ (i : x.LeftMoves), Exists fun j => HasEquiv.Equiv (x.moveLeft i) (y.mo …
      hr₁ : ∀ (i : x.RightMoves), Exists fun j => HasEquiv.Equiv (x.moveRight i) (y. …
      hl₂ : ∀ (j : y.LeftMoves), Exists fun i => HasEquiv.Equiv (x.moveLeft i) (y.mo …
      hr₂ : ∀ (j : y.RightMoves), Exists fun i => HasEquiv.Equiv (x.moveRight i) (y. …
      i : x.RightMoves
      j : y.RightMoves
      hj : HasEquiv.Equiv (x.moveRight i) (y.moveRight j)
      ⊢ Or (Exists fun i_1 => LE.le y ((x.moveRight i).moveLeft i_1)) (Exists fun j' …
    -/
    exact Or.inr ⟨j, Equiv.ge hj⟩
    /-
      🎉 no goals
    -/


theorem Equiv.of_equiv {x y : PGame} (L : x.LeftMoves ≃ y.LeftMoves)
    (R : x.RightMoves ≃ y.RightMoves) (hl : ∀ i, x.moveLeft i ≈ y.moveLeft (L i))
    (hr : ∀ j, x.moveRight j ≈ y.moveRight (R j)) : x ≈ y := by
  /-
    x y : SetTheory.PGame
    L : _root_.Equiv x.LeftMoves y.LeftMoves
    R : _root_.Equiv x.RightMoves y.RightMoves
    hl : ∀ (i : x.LeftMoves), HasEquiv.Equiv (x.moveLeft i) (y.moveLeft (L i))
    hr : ∀ (j : x.RightMoves), HasEquiv.Equiv (x.moveRight j) (y.moveRight (R j))
    ⊢ HasEquiv.Equiv x y
  -/
  apply Equiv.of_exists <;> intro i
  exacts [⟨_, hl i⟩, ⟨_, hr i⟩,
    ⟨_, by simpa using hl (L.symm i)⟩, ⟨_, by simpa using hr (R.symm i)⟩]


@[deprecated (since := "2024-09-26")] alias equiv_of_mk_equiv := Equiv.of_equiv


/-- The fuzzy, confused, or incomparable relation on pre-games.

If `x ‖ 0`, then the first player can always win `x`. -/
def Fuzzy (x y : PGame) : Prop :=
  x ⧏ y ∧ y ⧏ x


@[inherit_doc]
scoped infixl:50 " ‖ " => PGame.Fuzzy


@[symm]
theorem Fuzzy.swap {x y : PGame} : x ‖ y → y ‖ x :=
  And.symm


instance : IsSymm _ (· ‖ ·) :=
  ⟨fun _ _ => Fuzzy.swap⟩


theorem Fuzzy.swap_iff {x y : PGame} : x ‖ y ↔ y ‖ x :=
  ⟨Fuzzy.swap, Fuzzy.swap⟩


theorem fuzzy_irrefl (x : PGame) : ¬x ‖ x := fun h => lf_irrefl x h.1


instance : IsIrrefl _ (· ‖ ·) :=
  ⟨fuzzy_irrefl⟩


theorem lf_iff_lt_or_fuzzy {x y : PGame} : x ⧏ y ↔ x < y ∨ x ‖ y := by
  /-
    x y : SetTheory.PGame
    ⊢ Iff (x.LF y) (Or (LT.lt x y) (x.Fuzzy y))
  -/
  simp only [lt_iff_le_and_lf, Fuzzy, ← PGame.not_le]
  /-
    x y : SetTheory.PGame
    ⊢ Iff (Not (LE.le y x)) (Or (And (LE.le x y) (Not (LE.le y x))) (And (Not (LE. …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem lf_of_fuzzy {x y : PGame} (h : x ‖ y) : x ⧏ y :=
  lf_iff_lt_or_fuzzy.2 (Or.inr h)


alias Fuzzy.lf := lf_of_fuzzy


theorem lt_or_fuzzy_of_lf {x y : PGame} : x ⧏ y → x < y ∨ x ‖ y :=
  lf_iff_lt_or_fuzzy.1


theorem Fuzzy.not_equiv {x y : PGame} (h : x ‖ y) : ¬(x ≈ y) := fun h' => h'.1.not_gf h.2


theorem Fuzzy.not_equiv' {x y : PGame} (h : x ‖ y) : ¬(y ≈ x) := fun h' => h'.2.not_gf h.2


theorem not_fuzzy_of_le {x y : PGame} (h : x ≤ y) : ¬x ‖ y := fun h' => h'.2.not_ge h


theorem not_fuzzy_of_ge {x y : PGame} (h : y ≤ x) : ¬x ‖ y := fun h' => h'.1.not_ge h


theorem Equiv.not_fuzzy {x y : PGame} (h : x ≈ y) : ¬x ‖ y :=
  not_fuzzy_of_le h.1


theorem Equiv.not_fuzzy' {x y : PGame} (h : x ≈ y) : ¬y ‖ x :=
  not_fuzzy_of_le h.2


theorem fuzzy_congr {x₁ y₁ x₂ y₂ : PGame} (hx : x₁ ≈ x₂) (hy : y₁ ≈ y₂) : x₁ ‖ y₁ ↔ x₂ ‖ y₂ :=
                        /-
                          x₁ y₁ x₂ y₂ : SetTheory.PGame
                          hx : HasEquiv.Equiv x₁ x₂
                          hy : HasEquiv.Equiv y₁ y₂
                          ⊢ Iff (And (x₁.LF y₁) (y₁.LF x₁)) (And (x₂.LF y₂) (y₂.LF x₂))
                        -/
  show _ ∧ _ ↔ _ ∧ _ by rw [lf_congr hx hy, lf_congr hy hx]
                        /-
                          🎉 no goals
                        -/


theorem fuzzy_congr_imp {x₁ y₁ x₂ y₂ : PGame} (hx : x₁ ≈ x₂) (hy : y₁ ≈ y₂) : x₁ ‖ y₁ → x₂ ‖ y₂ :=
  (fuzzy_congr hx hy).1


theorem fuzzy_congr_left {x₁ x₂ y : PGame} (hx : x₁ ≈ x₂) : x₁ ‖ y ↔ x₂ ‖ y :=
  fuzzy_congr hx equiv_rfl


theorem fuzzy_congr_right {x y₁ y₂ : PGame} (hy : y₁ ≈ y₂) : x ‖ y₁ ↔ x ‖ y₂ :=
  fuzzy_congr equiv_rfl hy


@[trans]
theorem fuzzy_of_fuzzy_of_equiv {x y z : PGame} (h₁ : x ‖ y) (h₂ : y ≈ z) : x ‖ z :=
  (fuzzy_congr_right h₂).1 h₁


@[trans]
theorem fuzzy_of_equiv_of_fuzzy {x y z : PGame} (h₁ : x ≈ y) (h₂ : y ‖ z) : x ‖ z :=
  (fuzzy_congr_left h₁).2 h₂


/-- Exactly one of the following is true (although we don't prove this here). -/
theorem lt_or_equiv_or_gt_or_fuzzy (x y : PGame) : x < y ∨ (x ≈ y) ∨ y < x ∨ x ‖ y := by
  /-
    x y : SetTheory.PGame
    ⊢ Or (LT.lt x y) (Or (HasEquiv.Equiv x y) (Or (LT.lt y x) (x.Fuzzy y)))
  -/
  cases' le_or_gf x y with h₁ h₁ <;> cases' le_or_gf y x with h₂ h₂
    /-
      case inl.inl
      x y : SetTheory.PGame
      h₁ : LE.le x y
      h₂ : LE.le y x
      ⊢ Or (LT.lt x y) (Or (HasEquiv.Equiv x y) (Or (LT.lt y x) (x.Fuzzy y)))
    -/
  · right
    /-
      case inl.inl.h
      x y : SetTheory.PGame
      h₁ : LE.le x y
      h₂ : LE.le y x
      ⊢ Or (HasEquiv.Equiv x y) (Or (LT.lt y x) (x.Fuzzy y))
    -/
    left
    /-
      case inl.inl.h.h
      x y : SetTheory.PGame
      h₁ : LE.le x y
      h₂ : LE.le y x
      ⊢ HasEquiv.Equiv x y
    -/
    exact ⟨h₁, h₂⟩
    /-
      🎉 no goals
    -/
    /-
      case inl.inr
      x y : SetTheory.PGame
      h₁ : LE.le x y
      h₂ : x.LF y
      ⊢ Or (LT.lt x y) (Or (HasEquiv.Equiv x y) (Or (LT.lt y x) (x.Fuzzy y)))
    -/
  · left
    /-
      case inl.inr.h
      x y : SetTheory.PGame
      h₁ : LE.le x y
      h₂ : x.LF y
      ⊢ LT.lt x y
    -/
    exact ⟨h₁, h₂⟩
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      x y : SetTheory.PGame
      h₁ : y.LF x
      h₂ : LE.le y x
      ⊢ Or (LT.lt x y) (Or (HasEquiv.Equiv x y) (Or (LT.lt y x) (x.Fuzzy y)))
    -/
  · right
    /-
      case inr.inl.h
      x y : SetTheory.PGame
      h₁ : y.LF x
      h₂ : LE.le y x
      ⊢ Or (HasEquiv.Equiv x y) (Or (LT.lt y x) (x.Fuzzy y))
    -/
    right
    /-
      case inr.inl.h.h
      x y : SetTheory.PGame
      h₁ : y.LF x
      h₂ : LE.le y x
      ⊢ Or (LT.lt y x) (x.Fuzzy y)
    -/
    left
    /-
      case inr.inl.h.h.h
      x y : SetTheory.PGame
      h₁ : y.LF x
      h₂ : LE.le y x
      ⊢ LT.lt y x
    -/
    exact ⟨h₂, h₁⟩
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      x y : SetTheory.PGame
      h₁ : y.LF x
      h₂ : x.LF y
      ⊢ Or (LT.lt x y) (Or (HasEquiv.Equiv x y) (Or (LT.lt y x) (x.Fuzzy y)))
    -/
  · right
    /-
      case inr.inr.h
      x y : SetTheory.PGame
      h₁ : y.LF x
      h₂ : x.LF y
      ⊢ Or (HasEquiv.Equiv x y) (Or (LT.lt y x) (x.Fuzzy y))
    -/
    right
    /-
      case inr.inr.h.h
      x y : SetTheory.PGame
      h₁ : y.LF x
      h₂ : x.LF y
      ⊢ Or (LT.lt y x) (x.Fuzzy y)
    -/
    right
    /-
      case inr.inr.h.h.h
      x y : SetTheory.PGame
      h₁ : y.LF x
      h₂ : x.LF y
      ⊢ x.Fuzzy y
    -/
    exact ⟨h₂, h₁⟩
    /-
      🎉 no goals
    -/


theorem lt_or_equiv_or_gf (x y : PGame) : x < y ∨ (x ≈ y) ∨ y ⧏ x := by
  /-
    x y : SetTheory.PGame
    ⊢ Or (LT.lt x y) (Or (HasEquiv.Equiv x y) (y.LF x))
  -/
  rw [lf_iff_lt_or_fuzzy, Fuzzy.swap_iff]
  /-
    x y : SetTheory.PGame
    ⊢ Or (LT.lt x y) (Or (HasEquiv.Equiv x y) (Or (LT.lt y x) (x.Fuzzy y)))
  -/
  exact lt_or_equiv_or_gt_or_fuzzy x y
  /-
    🎉 no goals
  -/


/-- `Relabelling x y` says that `x` and `y` are really the same game, just dressed up differently.
Specifically, there is a bijection between the moves for Left in `x` and in `y`, and similarly
for Right, and under these bijections we inductively have `Relabelling`s for the consequent games.
-/
inductive Relabelling : PGame.{u} → PGame.{u} → Type (u + 1)
  |
  mk :
    ∀ {x y : PGame} (L : x.LeftMoves ≃ y.LeftMoves) (R : x.RightMoves ≃ y.RightMoves),
      (∀ i, Relabelling (x.moveLeft i) (y.moveLeft (L i))) →
        (∀ j, Relabelling (x.moveRight j) (y.moveRight (R j))) → Relabelling x y


@[inherit_doc]
scoped infixl:50 " ≡r " => PGame.Relabelling


/-- A constructor for relabellings swapping the equivalences. -/
def mk' (L : y.LeftMoves ≃ x.LeftMoves) (R : y.RightMoves ≃ x.RightMoves)
    (hL : ∀ i, x.moveLeft (L i) ≡r y.moveLeft i) (hR : ∀ j, x.moveRight (R j) ≡r y.moveRight j) :
    x ≡r y :=
                               /-
                                 xl xr : Type u
                                 x y : SetTheory.PGame
                                 L : _root_.Equiv y.LeftMoves x.LeftMoves
                                 R : _root_.Equiv y.RightMoves x.RightMoves
                                 hL : (i : y.LeftMoves) → (x.moveLeft (L i)).Relabelling (y.moveLeft i)
                                 hR : (j : y.RightMoves) → (x.moveRight (R j)).Relabelling (y.moveRight j)
                                 i : x.LeftMoves
                                 ⊢ (x.moveLeft i).Relabelling (y.moveLeft (L.symm i))
                               -/
                               /-
                                 🎉 no goals
                               -/
  ⟨L.symm, R.symm, fun i => by simpa using hL (L.symm i), fun j => by simpa using hR (R.symm j)⟩
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- The equivalence between left moves of `x` and `y` given by the relabelling. -/
def leftMovesEquiv : x ≡r y → x.LeftMoves ≃ y.LeftMoves
  | ⟨L,_, _,_⟩ => L


@[simp]
theorem mk_leftMovesEquiv {x y L R hL hR} : (@Relabelling.mk x y L R hL hR).leftMovesEquiv = L :=
  rfl


@[simp]
theorem mk'_leftMovesEquiv {x y L R hL hR} :
    (@Relabelling.mk' x y L R hL hR).leftMovesEquiv = L.symm :=
  rfl


/-- The equivalence between right moves of `x` and `y` given by the relabelling. -/
def rightMovesEquiv : x ≡r y → x.RightMoves ≃ y.RightMoves
  | ⟨_, R, _, _⟩ => R


@[simp]
theorem mk_rightMovesEquiv {x y L R hL hR} : (@Relabelling.mk x y L R hL hR).rightMovesEquiv = R :=
  rfl


@[simp]
theorem mk'_rightMovesEquiv {x y L R hL hR} :
    (@Relabelling.mk' x y L R hL hR).rightMovesEquiv = R.symm :=
  rfl


/-- A left move of `x` is a relabelling of a left move of `y`. -/
def moveLeft : ∀ (r : x ≡r y) (i : x.LeftMoves), x.moveLeft i ≡r y.moveLeft (r.leftMovesEquiv i)
  | ⟨_, _, hL, _⟩ => hL


/-- A left move of `y` is a relabelling of a left move of `x`. -/
def moveLeftSymm :
    ∀ (r : x ≡r y) (i : y.LeftMoves), x.moveLeft (r.leftMovesEquiv.symm i) ≡r y.moveLeft i
                            /-
                              xl xr : Type u
                              x y : SetTheory.PGame
                              L : _root_.Equiv x.LeftMoves y.LeftMoves
                              R : _root_.Equiv x.RightMoves y.RightMoves
                              hL : (i : x.LeftMoves) → (x.moveLeft i).Relabelling (y.moveLeft (L i))
                              hR : (j : x.RightMoves) → (x.moveRight j).Relabelling (y.moveRight (R j))
                              i : y.LeftMoves
                              ⊢ (x.moveLeft ((SetTheory.PGame.Relabelling.mk L R hL hR).leftMovesEquiv.symm  …
                            -/
  | ⟨L, R, hL, hR⟩, i => by simpa using hL (L.symm i)
                            /-
                              🎉 no goals
                            -/


/-- A right move of `x` is a relabelling of a right move of `y`. -/
def moveRight :
    ∀ (r : x ≡r y) (i : x.RightMoves), x.moveRight i ≡r y.moveRight (r.rightMovesEquiv i)
  | ⟨_, _, _, hR⟩ => hR


/-- A right move of `y` is a relabelling of a right move of `x`. -/
def moveRightSymm :
    ∀ (r : x ≡r y) (i : y.RightMoves), x.moveRight (r.rightMovesEquiv.symm i) ≡r y.moveRight i
                            /-
                              xl xr : Type u
                              x y : SetTheory.PGame
                              L : _root_.Equiv x.LeftMoves y.LeftMoves
                              R : _root_.Equiv x.RightMoves y.RightMoves
                              hL : (i : x.LeftMoves) → (x.moveLeft i).Relabelling (y.moveLeft (L i))
                              hR : (j : x.RightMoves) → (x.moveRight j).Relabelling (y.moveRight (R j))
                              i : y.RightMoves
                              ⊢ (x.moveRight ((SetTheory.PGame.Relabelling.mk L R hL hR).rightMovesEquiv.sym …
                            -/
  | ⟨L, R, hL, hR⟩, i => by simpa using hR (R.symm i)
                            /-
                              🎉 no goals
                            -/


/-- The identity relabelling. -/
@[refl]
def refl (x : PGame) : x ≡r x :=
  ⟨Equiv.refl _, Equiv.refl _, fun _ => refl _, fun _ => refl _⟩
termination_by x


instance (x : PGame) : Inhabited (x ≡r x) :=
  ⟨refl _⟩


/-- Flip a relabelling. -/
@[symm]
def symm : ∀ {x y : PGame}, x ≡r y → y ≡r x
  | _, _, ⟨L, R, hL, hR⟩ => mk' L R (fun i => (hL i).symm) fun j => (hR j).symm


theorem le {x y : PGame} (r : x ≡r y) : x ≤ y :=
  le_def.2
    ⟨fun i => Or.inl ⟨_, (r.moveLeft i).le⟩, fun j =>
      Or.inr ⟨_, (r.moveRightSymm j).le⟩⟩
termination_by x


theorem ge {x y : PGame} (r : x ≡r y) : y ≤ x :=
  r.symm.le


/-- A relabelling lets us prove equivalence of games. -/
theorem equiv (r : x ≡r y) : x ≈ y :=
  ⟨r.le, r.ge⟩


/-- Transitivity of relabelling. -/
@[trans]
def trans : ∀ {x y z : PGame}, x ≡r y → y ≡r z → x ≡r z
  | _, _, _, ⟨L₁, R₁, hL₁, hR₁⟩, ⟨L₂, R₂, hL₂, hR₂⟩ =>
    ⟨L₁.trans L₂, R₁.trans R₂, fun i => (hL₁ i).trans (hL₂ _), fun j => (hR₁ j).trans (hR₂ _)⟩


/-- Any game without left or right moves is a relabelling of 0. -/
def isEmpty (x : PGame) [IsEmpty x.LeftMoves] [IsEmpty x.RightMoves] : x ≡r 0 :=
  ⟨Equiv.equivPEmpty _, Equiv.equivOfIsEmpty _ _, isEmptyElim, isEmptyElim⟩


theorem Equiv.isEmpty (x : PGame) [IsEmpty x.LeftMoves] [IsEmpty x.RightMoves] : x ≈ 0 :=
  (Relabelling.isEmpty x).equiv


instance {x y : PGame} : Coe (x ≡r y) (x ≈ y) :=
  ⟨Relabelling.equiv⟩


/-- Replace the types indexing the next moves for Left and Right by equivalent types. -/
def relabel {x : PGame} {xl' xr'} (el : xl' ≃ x.LeftMoves) (er : xr' ≃ x.RightMoves) : PGame :=
  ⟨xl', xr', x.moveLeft ∘ el, x.moveRight ∘ er⟩


@[simp]
theorem relabel_moveLeft' {x : PGame} {xl' xr'} (el : xl' ≃ x.LeftMoves) (er : xr' ≃ x.RightMoves)
    (i : xl') : moveLeft (relabel el er) i = x.moveLeft (el i) :=
  rfl


theorem relabel_moveLeft {x : PGame} {xl' xr'} (el : xl' ≃ x.LeftMoves) (er : xr' ≃ x.RightMoves)
                                                                                  /-
                                                                                    x : SetTheory.PGame
                                                                                    xl' xr' : Type u_1
                                                                                    el : _root_.Equiv xl' x.LeftMoves
                                                                                    er : _root_.Equiv xr' x.RightMoves
                                                                                    i : x.LeftMoves
                                                                                    ⊢ Eq ((SetTheory.PGame.relabel el er).moveLeft (el.symm i)) (x.moveLeft i)
                                                                                  -/
    (i : x.LeftMoves) : moveLeft (relabel el er) (el.symm i) = x.moveLeft i := by simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[simp]
theorem relabel_moveRight' {x : PGame} {xl' xr'} (el : xl' ≃ x.LeftMoves) (er : xr' ≃ x.RightMoves)
    (j : xr') : moveRight (relabel el er) j = x.moveRight (er j) :=
  rfl


theorem relabel_moveRight {x : PGame} {xl' xr'} (el : xl' ≃ x.LeftMoves) (er : xr' ≃ x.RightMoves)
                                                                                     /-
                                                                                       x : SetTheory.PGame
                                                                                       xl' xr' : Type u_1
                                                                                       el : _root_.Equiv xl' x.LeftMoves
                                                                                       er : _root_.Equiv xr' x.RightMoves
                                                                                       j : x.RightMoves
                                                                                       ⊢ Eq ((SetTheory.PGame.relabel el er).moveRight (er.symm j)) (x.moveRight j)
                                                                                     -/
    (j : x.RightMoves) : moveRight (relabel el er) (er.symm j) = x.moveRight j := by simp
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


/-- The game obtained by relabelling the next moves is a relabelling of the original game. -/
def relabelRelabelling {x : PGame} {xl' xr'} (el : xl' ≃ x.LeftMoves) (er : xr' ≃ x.RightMoves) :
    x ≡r relabel el er :=
  -- Porting note: needed to add `rfl`
                                     /-
                                       xl xr : Type u
                                       x : SetTheory.PGame
                                       xl' xr' : Type ?u.100946
                                       el : _root_.Equiv xl' x.LeftMoves
                                       er : _root_.Equiv xr' x.RightMoves
                                       i : (SetTheory.PGame.relabel el er).LeftMoves
                                       ⊢ (x.moveLeft (el i)).Relabelling ((SetTheory.PGame.relabel el er).moveLeft i)
                                     -/
                                           /-
                                             🎉 no goals
                                           -/
  Relabelling.mk' el er (fun i => by simp; rfl) (fun j => by simp; rfl)
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- The negation of `{L | R}` is `{-R | -L}`. -/
def neg : PGame → PGame
  | ⟨l, r, L, R⟩ => ⟨r, l, fun i => neg (R i), fun i => neg (L i)⟩


instance : Neg PGame :=
  ⟨neg⟩


@[simp]
theorem neg_def {xl xr xL xR} : -mk xl xr xL xR = mk xr xl (fun j => -xR j) fun i => -xL i :=
  rfl


instance : InvolutiveNeg PGame :=
  { inferInstanceAs (Neg PGame) with
    neg_neg := fun x => by
      /-
        xl xr : Type u
        x : SetTheory.PGame
        ⊢ Eq (Neg.neg (Neg.neg x)) x
      -/
      induction' x with xl xr xL xR ihL ihR
      /-
        case mk
        xl✝ xr✝ : Type u
        xl xr : Type ?u.101543
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        ihL : ∀ (a : xl), Eq (Neg.neg (Neg.neg (xL a))) (xL a)
        ihR : ∀ (a : xr), Eq (Neg.neg (Neg.neg (xR a))) (xR a)
        ⊢ Eq (Neg.neg (Neg.neg (SetTheory.PGame.mk xl xr xL xR))) (SetTheory.PGame.mk  …
      -/
      simp_rw [neg_def, ihL, ihR] }
      /-
        🎉 no goals
      -/


instance : NegZeroClass PGame :=
  { inferInstanceAs (Zero PGame), inferInstanceAs (Neg PGame) with
    neg_zero := by
      /-
        xl xr : Type u
        ⊢ Eq (-0) 0
      -/
      dsimp [Zero.zero, Neg.neg, neg]
      /-
        xl xr : Type u
        ⊢ Eq (SetTheory.PGame.mk PEmpty.{?u.102003 + 1} PEmpty.{?u.102003 + 1} (fun i  …
      -/
                             /-
                               🎉 no goals
                             -/
      congr <;> funext i <;> cases i }
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem neg_ofLists (L R : List PGame) :
    -ofLists L R = ofLists (R.map fun x => -x) (L.map fun x => -x) := by
  /-
    L R : List SetTheory.PGame
    ⊢ Eq (Neg.neg (SetTheory.PGame.ofLists L R)) (SetTheory.PGame.ofLists (List.ma …
  -/
  simp only [ofLists, neg_def, List.getElem_map, mk.injEq, List.length_map, true_and]
  /-
    L R : List SetTheory.PGame
    ⊢ And (HEq (fun j => Neg.neg (GetElem.getElem R ↑j.down ⋯)) fun i => Neg.neg ( …
  -/
  constructor
  all_goals
    apply hfunext
    · simp
    · rintro ⟨⟨a, ha⟩⟩ ⟨⟨b, hb⟩⟩ h
      have :
        ∀ {m n} (_ : m = n) {b : ULift (Fin m)} {c : ULift (Fin n)} (_ : HEq b c),
          (b.down : ℕ) = ↑c.down := by
        rintro m n rfl b c
        simp only [heq_eq_eq]
        rintro rfl
        rfl
      simp only [heq_eq_eq]
      congr 5
      exact this (List.length_map _ _).symm h


theorem isOption_neg {x y : PGame} : IsOption x (-y) ↔ IsOption (-x) y := by
  /-
    x y : SetTheory.PGame
    ⊢ Iff (x.IsOption (Neg.neg y)) ((Neg.neg x).IsOption y)
  -/
  rw [isOption_iff, isOption_iff, or_comm]
  /-
    x y : SetTheory.PGame
    ⊢ Iff (Or (Exists fun i => Eq x ((Neg.neg y).moveRight i)) (Exists fun i => Eq …
  -/
  cases y
  /-
    case mk
    x : SetTheory.PGame
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    ⊢ Iff (Or (Exists fun i => Eq x ((Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝)).m …
  -/
  apply or_congr <;>
      /-
        case mk.h₁
        x : SetTheory.PGame
        α✝ β✝ : Type u_1
        a✝¹ : α✝ → SetTheory.PGame
        a✝ : β✝ → SetTheory.PGame
        ⊢ Iff (Exists fun i => Eq x ((Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝)).moveR …
      -/
      /-
        case mk.h₁.h
        x : SetTheory.PGame
        α✝ β✝ : Type u_1
        a✝¹ : α✝ → SetTheory.PGame
        a✝ : β✝ → SetTheory.PGame
        ⊢ ∀ (a : (Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝)).RightMoves), Iff (Eq x (( …
      -/
      /-
        case mk.h₁.h
        x : SetTheory.PGame
        α✝ β✝ : Type u_1
        a✝² : α✝ → SetTheory.PGame
        a✝¹ : β✝ → SetTheory.PGame
        a✝ : (Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝² a✝¹)).RightMoves
        ⊢ Iff (Eq x ((Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝² a✝¹)).moveRight a✝)) (Eq ( …
      -/
      /-
        case mk.h₁.h
        x : SetTheory.PGame
        α✝ β✝ : Type u_1
        a✝² : α✝ → SetTheory.PGame
        a✝¹ : β✝ → SetTheory.PGame
        a✝ : (Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝² a✝¹)).RightMoves
        ⊢ Iff (Eq x ((Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝² a✝¹)).moveRight a✝)) (Eq x …
      -/
      /-
        🎉 no goals
      -/
      /-
        case mk.h₂.h
        x : SetTheory.PGame
        α✝ β✝ : Type u_1
        a✝² : α✝ → SetTheory.PGame
        a✝¹ : β✝ → SetTheory.PGame
        a✝ : (Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝² a✝¹)).LeftMoves
        ⊢ Iff (Eq x ((Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝² a✝¹)).moveLeft a✝)) (Eq (N …
      -/
      rw [neg_eq_iff_eq_neg]
      /-
        case mk.h₂.h
        x : SetTheory.PGame
        α✝ β✝ : Type u_1
        a✝² : α✝ → SetTheory.PGame
        a✝¹ : β✝ → SetTheory.PGame
        a✝ : (Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝² a✝¹)).LeftMoves
        ⊢ Iff (Eq x ((Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝² a✝¹)).moveLeft a✝)) (Eq x  …
      -/
      rfl
      /-
        🎉 no goals
      -/


@[simp]
theorem isOption_neg_neg {x y : PGame} : IsOption (-x) (-y) ↔ IsOption x y := by
  /-
    x y : SetTheory.PGame
    ⊢ Iff ((Neg.neg x).IsOption (Neg.neg y)) (x.IsOption y)
  -/
  rw [isOption_neg, neg_neg]
  /-
    🎉 no goals
  -/


/-- Use `toLeftMovesNeg` to cast between these two types. -/
theorem leftMoves_neg : ∀ x : PGame, (-x).LeftMoves = x.RightMoves
  | ⟨_, _, _, _⟩ => rfl


/-- Use `toRightMovesNeg` to cast between these two types. -/
theorem rightMoves_neg : ∀ x : PGame, (-x).RightMoves = x.LeftMoves
  | ⟨_, _, _, _⟩ => rfl


/-- Turns a right move for `x` into a left move for `-x` and vice versa.

Even though these types are the same (not definitionally so), this is the preferred way to convert
between them. -/
def toLeftMovesNeg {x : PGame} : x.RightMoves ≃ (-x).LeftMoves :=
  Equiv.cast (leftMoves_neg x).symm


/-- Turns a left move for `x` into a right move for `-x` and vice versa.

Even though these types are the same (not definitionally so), this is the preferred way to convert
between them. -/
def toRightMovesNeg {x : PGame} : x.LeftMoves ≃ (-x).RightMoves :=
  Equiv.cast (rightMoves_neg x).symm


@[simp]
theorem moveLeft_neg {x : PGame} (i) :
    (-x).moveLeft i = -x.moveRight (toLeftMovesNeg.symm i) := by
  /-
    x : SetTheory.PGame
    i : (Neg.neg x).LeftMoves
    ⊢ Eq ((Neg.neg x).moveLeft i) (Neg.neg (x.moveRight (SetTheory.PGame.toLeftMov …
  -/
  cases x
  /-
    case mk
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    i : (Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝)).LeftMoves
    ⊢ Eq ((Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝)).moveLeft i) (Neg.neg ((SetTh …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated moveLeft_neg (since := "2024-10-30")]
alias moveLeft_neg' := moveLeft_neg


theorem moveLeft_neg_toLeftMovesNeg {x : PGame} (i) :
                                                            /-
                                                              x : SetTheory.PGame
                                                              i : x.RightMoves
                                                              ⊢ Eq ((Neg.neg x).moveLeft (SetTheory.PGame.toLeftMovesNeg i)) (Neg.neg (x.mov …
                                                            -/
    (-x).moveLeft (toLeftMovesNeg i) = -x.moveRight i := by simp
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
theorem moveRight_neg {x : PGame} (i) :
    (-x).moveRight i = -x.moveLeft (toRightMovesNeg.symm i) := by
  /-
    x : SetTheory.PGame
    i : (Neg.neg x).RightMoves
    ⊢ Eq ((Neg.neg x).moveRight i) (Neg.neg (x.moveLeft (SetTheory.PGame.toRightMo …
  -/
  cases x
  /-
    case mk
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    i : (Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝)).RightMoves
    ⊢ Eq ((Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝)).moveRight i) (Neg.neg ((SetT …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated moveRight_neg (since := "2024-10-30")]
alias moveRight_neg' := moveRight_neg


theorem moveRight_neg_toRightMovesNeg {x : PGame} (i) :
                                                             /-
                                                               x : SetTheory.PGame
                                                               i : x.LeftMoves
                                                               ⊢ Eq ((Neg.neg x).moveRight (SetTheory.PGame.toRightMovesNeg i)) (Neg.neg (x.m …
                                                             -/
    (-x).moveRight (toRightMovesNeg i) = -x.moveLeft i := by simp
                                                             /-
                                                               🎉 no goals
                                                             -/


@[deprecated moveRight_neg (since := "2024-10-30")]
theorem moveLeft_neg_symm {x : PGame} (i) :
                                                                  /-
                                                                    x : SetTheory.PGame
                                                                    i : (Neg.neg x).RightMoves
                                                                    ⊢ Eq (x.moveLeft (SetTheory.PGame.toRightMovesNeg.symm i)) (Neg.neg ((Neg.neg  …
                                                                  -/
    x.moveLeft (toRightMovesNeg.symm i) = -(-x).moveRight i := by simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[deprecated moveRight_neg (since := "2024-10-30")]
theorem moveLeft_neg_symm' {x : PGame} (i) :
                                                             /-
                                                               x : SetTheory.PGame
                                                               i : x.LeftMoves
                                                               ⊢ Eq (x.moveLeft i) (Neg.neg ((Neg.neg x).moveRight (SetTheory.PGame.toRightMo …
                                                             -/
    x.moveLeft i = -(-x).moveRight (toRightMovesNeg i) := by simp
                                                             /-
                                                               🎉 no goals
                                                             -/


@[deprecated moveLeft_neg (since := "2024-10-30")]
theorem moveRight_neg_symm {x : PGame} (i) :
                                                                 /-
                                                                   x : SetTheory.PGame
                                                                   i : (Neg.neg x).LeftMoves
                                                                   ⊢ Eq (x.moveRight (SetTheory.PGame.toLeftMovesNeg.symm i)) (Neg.neg ((Neg.neg  …
                                                                 -/
    x.moveRight (toLeftMovesNeg.symm i) = -(-x).moveLeft i := by simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[deprecated moveLeft_neg (since := "2024-10-30")]
theorem moveRight_neg_symm' {x : PGame} (i) :
                                                            /-
                                                              x : SetTheory.PGame
                                                              i : x.RightMoves
                                                              ⊢ Eq (x.moveRight i) (Neg.neg ((Neg.neg x).moveLeft (SetTheory.PGame.toLeftMov …
                                                            -/
    x.moveRight i = -(-x).moveLeft (toLeftMovesNeg i) := by simp
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp] theorem neg_identical_neg_iff : ∀ {x y : PGame.{u}}, -x ≡ -y ↔ x ≡ y
  | mk xl xr xL xR, mk yl yr yL yR => by
    /-
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      ⊢ Iff ((Neg.neg (SetTheory.PGame.mk xl xr xL xR)).Identical (Neg.neg (SetTheor …
    -/
    rw [neg_def, identical_iff, identical_iff, ← neg_def, and_comm]
    /-
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      ⊢ Iff (And (Relator.BiTotal fun x1 x2 => ((Neg.neg (SetTheory.PGame.mk xl xr x …
    -/
    simp only [neg_def, rightMoves_mk, moveRight_mk, leftMoves_mk, moveLeft_mk]
    /-
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      ⊢ Iff (And (Relator.BiTotal fun x1 x2 => (Neg.neg (xL x1)).Identical (Neg.neg  …
    -/
    apply and_congr <;>
      /-
        case h₁
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        ⊢ Iff (Relator.BiTotal fun x1 x2 => (Neg.neg (xL x1)).Identical (Neg.neg (yL x …
      -/
        /-
          case h₁.mp
          xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          ⊢ (Relator.BiTotal fun x1 x2 => (Neg.neg (xL x1)).Identical (Neg.neg (yL x2))) …
        -/
        /-
          case h₁.mp
          xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          ⊢ (Relator.BiTotal fun x1 x2 => (xL x1).Identical (yL x2)) → Relator.BiTotal f …
        -/
        /-
          🎉 no goals
        -/
        /-
          case h₁.mpr
          xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          ⊢ (Relator.BiTotal fun x1 x2 => (xL x1).Identical (yL x2)) → Relator.BiTotal f …
        -/
        /-
          case h₁.mpr
          xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          ⊢ (Relator.BiTotal fun x1 x2 => (Neg.neg (xL x1)).Identical (Neg.neg (yL x2))) …
        -/
        /-
          🎉 no goals
        -/
        simp only [imp_self]
        /-
          🎉 no goals
        -/
        /-
          case h₂.mpr
          xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          ⊢ (Relator.BiTotal fun x1 x2 => (xR x1).Identical (yR x2)) → Relator.BiTotal f …
        -/
      · conv in (_ ≡ _) => rw [← neg_identical_neg_iff]
        /-
          case h₂.mpr
          xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          ⊢ (Relator.BiTotal fun x1 x2 => (Neg.neg (xR x1)).Identical (Neg.neg (yR x2))) …
        -/
        simp only [imp_self]
        /-
          🎉 no goals
        -/
termination_by x y => (x, y)


theorem Identical.neg {x y : PGame} : x ≡ y ↔ -x ≡ -y :=
  neg_identical_neg_iff.symm


/-- If `x` has the same moves as `y`, then `-x` has the same moves as `-y`. -/
def Relabelling.negCongr : ∀ {x y : PGame}, x ≡r y → -x ≡r -y
  | ⟨_, _, _, _⟩, ⟨_, _, _, _⟩, ⟨L, R, hL, hR⟩ =>
    ⟨R, L, fun j => (hR j).negCongr, fun i => (hL i).negCongr⟩


private theorem neg_le_lf_neg_iff : ∀ {x y : PGame.{u}}, (-y ≤ -x ↔ x ≤ y) ∧ (-y ⧏ -x ↔ x ⧏ y)
  | mk xl xr xL xR, mk yl yr yL yR => by
    /-
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      ⊢ And (Iff (LE.le (Neg.neg (SetTheory.PGame.mk yl yr yL yR)) (Neg.neg (SetTheo …
    -/
    simp_rw [neg_def, mk_le_mk, mk_lf_mk, ← neg_def]
    /-
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      ⊢ And (Iff (And (∀ (i : yr), (Neg.neg (yR i)).LF (Neg.neg (SetTheory.PGame.mk  …
    -/
    constructor
      /-
        case left
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        ⊢ Iff (And (∀ (i : yr), (Neg.neg (yR i)).LF (Neg.neg (SetTheory.PGame.mk xl xr …
      -/
    · rw [and_comm]
      /-
        case left
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        ⊢ Iff (And (∀ (j : xl), (Neg.neg (SetTheory.PGame.mk yl yr yL yR)).LF (Neg.neg …
      -/
                          /-
                            🎉 no goals
                          -/
      apply and_congr <;> exact forall_congr' fun _ => neg_le_lf_neg_iff.2
                          /-
                            🎉 no goals
                          -/
      /-
        case right
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        ⊢ Iff (Or (Exists fun i => LE.le (Neg.neg (SetTheory.PGame.mk yl yr yL yR)) (N …
      -/
    · rw [or_comm]
      /-
        case right
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        ⊢ Iff (Or (Exists fun j => LE.le (Neg.neg (yL j)) (Neg.neg (SetTheory.PGame.mk …
      -/
                         /-
                           🎉 no goals
                         -/
      apply or_congr <;> exact exists_congr fun _ => neg_le_lf_neg_iff.1
                         /-
                           🎉 no goals
                         -/
termination_by x y => (x, y)


@[simp]
theorem neg_le_neg_iff {x y : PGame} : -y ≤ -x ↔ x ≤ y :=
  neg_le_lf_neg_iff.1


@[simp]
theorem neg_lf_neg_iff {x y : PGame} : -y ⧏ -x ↔ x ⧏ y :=
  neg_le_lf_neg_iff.2


@[simp]
theorem neg_lt_neg_iff {x y : PGame} : -y < -x ↔ x < y := by
  /-
    x y : SetTheory.PGame
    ⊢ Iff (LT.lt (Neg.neg y) (Neg.neg x)) (LT.lt x y)
  -/
  rw [lt_iff_le_and_lf, lt_iff_le_and_lf, neg_le_neg_iff, neg_lf_neg_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem neg_equiv_neg_iff {x y : PGame} : (-x ≈ -y) ↔ (x ≈ y) := by
  /-
    x y : SetTheory.PGame
    ⊢ Iff (HasEquiv.Equiv (Neg.neg x) (Neg.neg y)) (HasEquiv.Equiv x y)
  -/
  show Equiv (-x) (-y) ↔ Equiv x y
  /-
    x y : SetTheory.PGame
    ⊢ Iff ((Neg.neg x).Equiv (Neg.neg y)) (x.Equiv y)
  -/
  rw [Equiv, Equiv, neg_le_neg_iff, neg_le_neg_iff, and_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem neg_fuzzy_neg_iff {x y : PGame} : -x ‖ -y ↔ x ‖ y := by
  /-
    x y : SetTheory.PGame
    ⊢ Iff ((Neg.neg x).Fuzzy (Neg.neg y)) (x.Fuzzy y)
  -/
  rw [Fuzzy, Fuzzy, neg_lf_neg_iff, neg_lf_neg_iff, and_comm]
  /-
    🎉 no goals
  -/


                                                         /-
                                                           x y : SetTheory.PGame
                                                           ⊢ Iff (LE.le (Neg.neg y) x) (LE.le (Neg.neg x) y)
                                                         -/
theorem neg_le_iff {x y : PGame} : -y ≤ x ↔ -x ≤ y := by rw [← neg_neg x, neg_le_neg_iff, neg_neg]
                                                         /-
                                                           🎉 no goals
                                                         -/


                                                         /-
                                                           x y : SetTheory.PGame
                                                           ⊢ Iff ((Neg.neg y).LF x) ((Neg.neg x).LF y)
                                                         -/
theorem neg_lf_iff {x y : PGame} : -y ⧏ x ↔ -x ⧏ y := by rw [← neg_neg x, neg_lf_neg_iff, neg_neg]
                                                         /-
                                                           🎉 no goals
                                                         -/


                                                         /-
                                                           x y : SetTheory.PGame
                                                           ⊢ Iff (LT.lt (Neg.neg y) x) (LT.lt (Neg.neg x) y)
                                                         -/
theorem neg_lt_iff {x y : PGame} : -y < x ↔ -x < y := by rw [← neg_neg x, neg_lt_neg_iff, neg_neg]
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem neg_equiv_iff {x y : PGame} : (-x ≈ y) ↔ (x ≈ -y) := by
  /-
    x y : SetTheory.PGame
    ⊢ Iff (HasEquiv.Equiv (Neg.neg x) y) (HasEquiv.Equiv x (Neg.neg y))
  -/
  rw [← neg_neg y, neg_equiv_neg_iff, neg_neg]
  /-
    🎉 no goals
  -/


theorem neg_fuzzy_iff {x y : PGame} : -x ‖ y ↔ x ‖ -y := by
  /-
    x y : SetTheory.PGame
    ⊢ Iff ((Neg.neg x).Fuzzy y) (x.Fuzzy (Neg.neg y))
  -/
  rw [← neg_neg y, neg_fuzzy_neg_iff, neg_neg]
  /-
    🎉 no goals
  -/


                                                         /-
                                                           x y : SetTheory.PGame
                                                           ⊢ Iff (LE.le y (Neg.neg x)) (LE.le x (Neg.neg y))
                                                         -/
theorem le_neg_iff {x y : PGame} : y ≤ -x ↔ x ≤ -y := by rw [← neg_neg x, neg_le_neg_iff, neg_neg]
                                                         /-
                                                           🎉 no goals
                                                         -/


                                                         /-
                                                           x y : SetTheory.PGame
                                                           ⊢ Iff (y.LF (Neg.neg x)) (x.LF (Neg.neg y))
                                                         -/
theorem lf_neg_iff {x y : PGame} : y ⧏ -x ↔ x ⧏ -y := by rw [← neg_neg x, neg_lf_neg_iff, neg_neg]
                                                         /-
                                                           🎉 no goals
                                                         -/


                                                         /-
                                                           x y : SetTheory.PGame
                                                           ⊢ Iff (LT.lt y (Neg.neg x)) (LT.lt x (Neg.neg y))
                                                         -/
theorem lt_neg_iff {x y : PGame} : y < -x ↔ x < -y := by rw [← neg_neg x, neg_lt_neg_iff, neg_neg]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
                                                           /-
                                                             x : SetTheory.PGame
                                                             ⊢ Iff (LE.le (Neg.neg x) 0) (LE.le 0 x)
                                                           -/
theorem neg_le_zero_iff {x : PGame} : -x ≤ 0 ↔ 0 ≤ x := by rw [neg_le_iff, neg_zero]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
                                                           /-
                                                             x : SetTheory.PGame
                                                             ⊢ Iff (LE.le 0 (Neg.neg x)) (LE.le x 0)
                                                           -/
theorem zero_le_neg_iff {x : PGame} : 0 ≤ -x ↔ x ≤ 0 := by rw [le_neg_iff, neg_zero]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
                                                           /-
                                                             x : SetTheory.PGame
                                                             ⊢ Iff ((Neg.neg x).LF 0) (SetTheory.PGame.LF 0 x)
                                                           -/
theorem neg_lf_zero_iff {x : PGame} : -x ⧏ 0 ↔ 0 ⧏ x := by rw [neg_lf_iff, neg_zero]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
                                                           /-
                                                             x : SetTheory.PGame
                                                             ⊢ Iff (SetTheory.PGame.LF 0 (Neg.neg x)) (x.LF 0)
                                                           -/
theorem zero_lf_neg_iff {x : PGame} : 0 ⧏ -x ↔ x ⧏ 0 := by rw [lf_neg_iff, neg_zero]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
                                                           /-
                                                             x : SetTheory.PGame
                                                             ⊢ Iff (LT.lt (Neg.neg x) 0) (LT.lt 0 x)
                                                           -/
theorem neg_lt_zero_iff {x : PGame} : -x < 0 ↔ 0 < x := by rw [neg_lt_iff, neg_zero]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
                                                           /-
                                                             x : SetTheory.PGame
                                                             ⊢ Iff (LT.lt 0 (Neg.neg x)) (LT.lt x 0)
                                                           -/
theorem zero_lt_neg_iff {x : PGame} : 0 < -x ↔ x < 0 := by rw [lt_neg_iff, neg_zero]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
                                                                  /-
                                                                    x : SetTheory.PGame
                                                                    ⊢ Iff (HasEquiv.Equiv (Neg.neg x) 0) (HasEquiv.Equiv x 0)
                                                                  -/
theorem neg_equiv_zero_iff {x : PGame} : (-x ≈ 0) ↔ (x ≈ 0) := by rw [neg_equiv_iff, neg_zero]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
                                                              /-
                                                                x : SetTheory.PGame
                                                                ⊢ Iff ((Neg.neg x).Fuzzy 0) (x.Fuzzy 0)
                                                              -/
theorem neg_fuzzy_zero_iff {x : PGame} : -x ‖ 0 ↔ x ‖ 0 := by rw [neg_fuzzy_iff, neg_zero]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
                                                                  /-
                                                                    x : SetTheory.PGame
                                                                    ⊢ Iff (HasEquiv.Equiv 0 (Neg.neg x)) (HasEquiv.Equiv 0 x)
                                                                  -/
theorem zero_equiv_neg_iff {x : PGame} : (0 ≈ -x) ↔ (0 ≈ x) := by rw [← neg_equiv_iff, neg_zero]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
                                                              /-
                                                                x : SetTheory.PGame
                                                                ⊢ Iff (SetTheory.PGame.Fuzzy 0 (Neg.neg x)) (SetTheory.PGame.Fuzzy 0 x)
                                                              -/
theorem zero_fuzzy_neg_iff {x : PGame} : 0 ‖ -x ↔ 0 ‖ x := by rw [← neg_fuzzy_iff, neg_zero]
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- The sum of `x = {xL | xR}` and `y = {yL | yR}` is `{xL + y, x + yL | xR + y, x + yR}`. -/
instance : Add PGame.{u} :=
  ⟨fun x y => by
    /-
      xl xr : Type u
      x y : SetTheory.PGame
      ⊢ SetTheory.PGame
    -/
    induction x generalizing y with | mk xl xr _ _ IHxl IHxr => _
    /-
      case mk
      xl✝ xr✝ xl xr : Type u
      a✝¹ : xl → SetTheory.PGame
      a✝ : xr → SetTheory.PGame
      IHxl : xl → SetTheory.PGame → SetTheory.PGame
      IHxr : xr → SetTheory.PGame → SetTheory.PGame
      y : SetTheory.PGame
      ⊢ SetTheory.PGame
    -/
    induction y with | mk yl yr yL yR IHyl IHyr => _
    /-
      case mk.mk
      xl✝ xr✝ xl xr : Type u
      a✝¹ : xl → SetTheory.PGame
      a✝ : xr → SetTheory.PGame
      IHxl : xl → SetTheory.PGame → SetTheory.PGame
      IHxr : xr → SetTheory.PGame → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      IHyl : yl → SetTheory.PGame
      IHyr : yr → SetTheory.PGame
      ⊢ SetTheory.PGame
    -/
    have y := mk yl yr yL yR
    /-
      case mk.mk
      xl✝ xr✝ xl xr : Type u
      a✝¹ : xl → SetTheory.PGame
      a✝ : xr → SetTheory.PGame
      IHxl : xl → SetTheory.PGame → SetTheory.PGame
      IHxr : xr → SetTheory.PGame → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      IHyl : yl → SetTheory.PGame
      IHyr : yr → SetTheory.PGame
      y : SetTheory.PGame
      ⊢ SetTheory.PGame
    -/
    refine ⟨xl ⊕ yl, xr ⊕ yr, Sum.rec ?_ ?_, Sum.rec ?_ ?_⟩
      /-
        case mk.mk.refine_1
        xl✝ xr✝ xl xr : Type u
        a✝¹ : xl → SetTheory.PGame
        a✝ : xr → SetTheory.PGame
        IHxl : xl → SetTheory.PGame → SetTheory.PGame
        IHxr : xr → SetTheory.PGame → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        IHyl : yl → SetTheory.PGame
        IHyr : yr → SetTheory.PGame
        y : SetTheory.PGame
        ⊢ xl → SetTheory.PGame
      -/
    · exact fun i => IHxl i y
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.refine_2
        xl✝ xr✝ xl xr : Type u
        a✝¹ : xl → SetTheory.PGame
        a✝ : xr → SetTheory.PGame
        IHxl : xl → SetTheory.PGame → SetTheory.PGame
        IHxr : xr → SetTheory.PGame → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        IHyl : yl → SetTheory.PGame
        IHyr : yr → SetTheory.PGame
        y : SetTheory.PGame
        ⊢ yl → SetTheory.PGame
      -/
    · exact IHyl
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.refine_3
        xl✝ xr✝ xl xr : Type u
        a✝¹ : xl → SetTheory.PGame
        a✝ : xr → SetTheory.PGame
        IHxl : xl → SetTheory.PGame → SetTheory.PGame
        IHxr : xr → SetTheory.PGame → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        IHyl : yl → SetTheory.PGame
        IHyr : yr → SetTheory.PGame
        y : SetTheory.PGame
        ⊢ xr → SetTheory.PGame
      -/
    · exact fun i => IHxr i y
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.refine_4
        xl✝ xr✝ xl xr : Type u
        a✝¹ : xl → SetTheory.PGame
        a✝ : xr → SetTheory.PGame
        IHxl : xl → SetTheory.PGame → SetTheory.PGame
        IHxr : xr → SetTheory.PGame → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        IHyl : yl → SetTheory.PGame
        IHyr : yr → SetTheory.PGame
        y : SetTheory.PGame
        ⊢ yr → SetTheory.PGame
      -/
    · exact IHyr⟩
      /-
        🎉 no goals
      -/


/-- The pre-game `((0 + 1) + ⋯) + 1`.

Note that this is **not** the usual recursive definition `n = {0, 1, … | }`. For instance,
`2 = 0 + 1 + 1 = {0 + 0 + 1, 0 + 1 + 0 | }` does not contain any left option equivalent to `0`. For
an implementation of said definition, see `Ordinal.toPGame`. For the proof that these games are
equivalent, see `Ordinal.toPGame_natCast`. -/
instance : NatCast PGame :=
  ⟨Nat.unaryCast⟩


@[simp]
protected theorem nat_succ (n : ℕ) : ((n + 1 : ℕ) : PGame) = n + 1 :=
  rfl


instance isEmpty_leftMoves_add (x y : PGame.{u}) [IsEmpty x.LeftMoves] [IsEmpty y.LeftMoves] :
    IsEmpty (x + y).LeftMoves := by
  /-
    xl xr : Type u
    x y : SetTheory.PGame
    inst✝¹ : IsEmpty x.LeftMoves
    inst✝ : IsEmpty y.LeftMoves
    ⊢ IsEmpty (HAdd.hAdd x y).LeftMoves
  -/
  cases x
  /-
    case mk
    xl xr : Type u
    y : SetTheory.PGame
    inst✝¹ : IsEmpty y.LeftMoves
    α✝ β✝ : Type u
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    inst✝ : IsEmpty (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).LeftMoves
    ⊢ IsEmpty (HAdd.hAdd (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝) y).LeftMoves
  -/
  cases y
  /-
    case mk.mk
    xl xr α✝¹ β✝¹ : Type u
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    inst✝¹ : IsEmpty (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²).LeftMoves
    α✝ β✝ : Type u
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    inst✝ : IsEmpty (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).LeftMoves
    ⊢ IsEmpty (HAdd.hAdd (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²) (SetTheory.PGame.mk  …
  -/
  apply isEmpty_sum.2 ⟨_, _⟩
  /-
    xl xr α✝¹ β✝¹ : Type u
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    inst✝¹ : IsEmpty (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²).LeftMoves
    α✝ β✝ : Type u
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    inst✝ : IsEmpty (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).LeftMoves
    ⊢ IsEmpty α✝¹
  -/
  assumption'
  /-
    🎉 no goals
  -/


instance isEmpty_rightMoves_add (x y : PGame.{u}) [IsEmpty x.RightMoves] [IsEmpty y.RightMoves] :
    IsEmpty (x + y).RightMoves := by
  /-
    xl xr : Type u
    x y : SetTheory.PGame
    inst✝¹ : IsEmpty x.RightMoves
    inst✝ : IsEmpty y.RightMoves
    ⊢ IsEmpty (HAdd.hAdd x y).RightMoves
  -/
  cases x
  /-
    case mk
    xl xr : Type u
    y : SetTheory.PGame
    inst✝¹ : IsEmpty y.RightMoves
    α✝ β✝ : Type u
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    inst✝ : IsEmpty (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).RightMoves
    ⊢ IsEmpty (HAdd.hAdd (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝) y).RightMoves
  -/
  cases y
  /-
    case mk.mk
    xl xr α✝¹ β✝¹ : Type u
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    inst✝¹ : IsEmpty (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²).RightMoves
    α✝ β✝ : Type u
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    inst✝ : IsEmpty (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).RightMoves
    ⊢ IsEmpty (HAdd.hAdd (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²) (SetTheory.PGame.mk  …
  -/
  apply isEmpty_sum.2 ⟨_, _⟩
  /-
    xl xr α✝¹ β✝¹ : Type u
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    inst✝¹ : IsEmpty (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²).RightMoves
    α✝ β✝ : Type u
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    inst✝ : IsEmpty (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).RightMoves
    ⊢ IsEmpty β✝¹
  -/
  assumption'
  /-
    🎉 no goals
  -/


/-- `x + 0` has exactly the same moves as `x`. -/
def addZeroRelabelling : ∀ x : PGame.{u}, x + 0 ≡r x
  | ⟨xl, xr, xL, xR⟩ => by
    /-
      xl✝ xr✝ xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      ⊢ (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) 0).Relabelling (SetTheory.PGame. …
    -/
    refine ⟨Equiv.sumEmpty xl PEmpty, Equiv.sumEmpty xr PEmpty, ?_, ?_⟩ <;> rintro (⟨i⟩ | ⟨⟨⟩⟩) <;>
      /-
        case refine_1.inl
        xl✝ xr✝ xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        i : xl
        ⊢ ((HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) 0).moveLeft (Sum.inl i)).Relabe …
      -/
      /-
        🎉 no goals
      -/
      apply addZeroRelabelling
      /-
        🎉 no goals
      -/
termination_by x => x


/-- `x + 0` is equivalent to `x`. -/
theorem add_zero_equiv (x : PGame.{u}) : x + 0 ≈ x :=
  (addZeroRelabelling x).equiv


/-- `0 + x` has exactly the same moves as `x`. -/
def zeroAddRelabelling : ∀ x : PGame.{u}, 0 + x ≡r x
  | ⟨xl, xr, xL, xR⟩ => by
    /-
      xl✝ xr✝ xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      ⊢ (HAdd.hAdd 0 (SetTheory.PGame.mk xl xr xL xR)).Relabelling (SetTheory.PGame. …
    -/
    refine ⟨Equiv.emptySum PEmpty xl, Equiv.emptySum PEmpty xr, ?_, ?_⟩ <;> rintro (⟨⟨⟩⟩ | ⟨i⟩) <;>
      /-
        case refine_1.inr
        xl✝ xr✝ xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        i : xl
        ⊢ ((HAdd.hAdd 0 (SetTheory.PGame.mk xl xr xL xR)).moveLeft (Sum.inr i)).Relabe …
      -/
      /-
        🎉 no goals
      -/
      apply zeroAddRelabelling
      /-
        🎉 no goals
      -/


/-- `0 + x` is equivalent to `x`. -/
theorem zero_add_equiv (x : PGame.{u}) : 0 + x ≈ x :=
  (zeroAddRelabelling x).equiv


/-- Use `toLeftMovesAdd` to cast between these two types. -/
theorem leftMoves_add : ∀ x y : PGame.{u}, (x + y).LeftMoves = (x.LeftMoves ⊕ y.LeftMoves)
  | ⟨_, _, _, _⟩, ⟨_, _, _, _⟩ => rfl


/-- Use `toRightMovesAdd` to cast between these two types. -/
theorem rightMoves_add : ∀ x y : PGame.{u}, (x + y).RightMoves = (x.RightMoves ⊕ y.RightMoves)
  | ⟨_, _, _, _⟩, ⟨_, _, _, _⟩ => rfl


/-- Converts a left move for `x` or `y` into a left move for `x + y` and vice versa.

Even though these types are the same (not definitionally so), this is the preferred way to convert
between them. -/
def toLeftMovesAdd {x y : PGame} : x.LeftMoves ⊕ y.LeftMoves ≃ (x + y).LeftMoves :=
  Equiv.cast (leftMoves_add x y).symm


/-- Converts a right move for `x` or `y` into a right move for `x + y` and vice versa.

Even though these types are the same (not definitionally so), this is the preferred way to convert
between them. -/
def toRightMovesAdd {x y : PGame} : x.RightMoves ⊕ y.RightMoves ≃ (x + y).RightMoves :=
  Equiv.cast (rightMoves_add x y).symm


@[simp]
theorem mk_add_moveLeft_inl {xl xr yl yr} {xL xR yL yR} {i} :
    (mk xl xr xL xR + mk yl yr yL yR).moveLeft (Sum.inl i) =
      (mk xl xr xL xR).moveLeft i + mk yl yr yL yR :=
  rfl


@[simp]
theorem add_moveLeft_inl {x : PGame} (y : PGame) (i) :
    (x + y).moveLeft (toLeftMovesAdd (Sum.inl i)) = x.moveLeft i + y := by
  /-
    x y : SetTheory.PGame
    i : x.LeftMoves
    ⊢ Eq ((HAdd.hAdd x y).moveLeft (SetTheory.PGame.toLeftMovesAdd (Sum.inl i))) ( …
  -/
  cases x
  /-
    case mk
    y : SetTheory.PGame
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    i : (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).LeftMoves
    ⊢ Eq ((HAdd.hAdd (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝) y).moveLeft (SetTheory.PGam …
  -/
  cases y
  /-
    case mk.mk
    α✝¹ β✝¹ : Type u_1
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    i : (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²).LeftMoves
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    ⊢ Eq ((HAdd.hAdd (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²) (SetTheory.PGame.mk α✝ β …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_add_moveRight_inl {xl xr yl yr} {xL xR yL yR} {i} :
    (mk xl xr xL xR + mk yl yr yL yR).moveRight (Sum.inl i) =
      (mk xl xr xL xR).moveRight i + mk yl yr yL yR :=
  rfl


@[simp]
theorem add_moveRight_inl {x : PGame} (y : PGame) (i) :
    (x + y).moveRight (toRightMovesAdd (Sum.inl i)) = x.moveRight i + y := by
  /-
    x y : SetTheory.PGame
    i : x.RightMoves
    ⊢ Eq ((HAdd.hAdd x y).moveRight (SetTheory.PGame.toRightMovesAdd (Sum.inl i))) …
  -/
  cases x
  /-
    case mk
    y : SetTheory.PGame
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    i : (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).RightMoves
    ⊢ Eq ((HAdd.hAdd (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝) y).moveRight (SetTheory.PGa …
  -/
  cases y
  /-
    case mk.mk
    α✝¹ β✝¹ : Type u_1
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    i : (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²).RightMoves
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    ⊢ Eq ((HAdd.hAdd (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²) (SetTheory.PGame.mk α✝ β …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_add_moveLeft_inr {xl xr yl yr} {xL xR yL yR} {i} :
    (mk xl xr xL xR + mk yl yr yL yR).moveLeft (Sum.inr i) =
      mk xl xr xL xR + (mk yl yr yL yR).moveLeft i :=
  rfl


@[simp]
theorem add_moveLeft_inr (x : PGame) {y : PGame} (i) :
    (x + y).moveLeft (toLeftMovesAdd (Sum.inr i)) = x + y.moveLeft i := by
  /-
    x y : SetTheory.PGame
    i : y.LeftMoves
    ⊢ Eq ((HAdd.hAdd x y).moveLeft (SetTheory.PGame.toLeftMovesAdd (Sum.inr i))) ( …
  -/
  cases x
  /-
    case mk
    y : SetTheory.PGame
    i : y.LeftMoves
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    ⊢ Eq ((HAdd.hAdd (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝) y).moveLeft (SetTheory.PGam …
  -/
  cases y
  /-
    case mk.mk
    α✝¹ β✝¹ : Type u_1
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    i : (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).LeftMoves
    ⊢ Eq ((HAdd.hAdd (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²) (SetTheory.PGame.mk α✝ β …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_add_moveRight_inr {xl xr yl yr} {xL xR yL yR} {i} :
    (mk xl xr xL xR + mk yl yr yL yR).moveRight (Sum.inr i) =
      mk xl xr xL xR + (mk yl yr yL yR).moveRight i :=
  rfl


@[simp]
theorem add_moveRight_inr (x : PGame) {y : PGame} (i) :
    (x + y).moveRight (toRightMovesAdd (Sum.inr i)) = x + y.moveRight i := by
  /-
    x y : SetTheory.PGame
    i : y.RightMoves
    ⊢ Eq ((HAdd.hAdd x y).moveRight (SetTheory.PGame.toRightMovesAdd (Sum.inr i))) …
  -/
  cases x
  /-
    case mk
    y : SetTheory.PGame
    i : y.RightMoves
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    ⊢ Eq ((HAdd.hAdd (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝) y).moveRight (SetTheory.PGa …
  -/
  cases y
  /-
    case mk.mk
    α✝¹ β✝¹ : Type u_1
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    i : (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝).RightMoves
    ⊢ Eq ((HAdd.hAdd (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²) (SetTheory.PGame.mk α✝ β …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Case on possible left moves of `x + y`. -/
theorem leftMoves_add_cases {x y : PGame} (k) {P : (x + y).LeftMoves → Prop}
    (hl : ∀ i, P <| toLeftMovesAdd (Sum.inl i)) (hr : ∀ i, P <| toLeftMovesAdd (Sum.inr i)) :
    P k := by
  /-
    x y : SetTheory.PGame
    k : (HAdd.hAdd x y).LeftMoves
    P : (HAdd.hAdd x y).LeftMoves → Prop
    hl : ∀ (i : x.LeftMoves), P (SetTheory.PGame.toLeftMovesAdd (Sum.inl i))
    hr : ∀ (i : y.LeftMoves), P (SetTheory.PGame.toLeftMovesAdd (Sum.inr i))
    ⊢ P k
  -/
  rw [← toLeftMovesAdd.apply_symm_apply k]
  /-
    x y : SetTheory.PGame
    k : (HAdd.hAdd x y).LeftMoves
    P : (HAdd.hAdd x y).LeftMoves → Prop
    hl : ∀ (i : x.LeftMoves), P (SetTheory.PGame.toLeftMovesAdd (Sum.inl i))
    hr : ∀ (i : y.LeftMoves), P (SetTheory.PGame.toLeftMovesAdd (Sum.inr i))
    ⊢ P (SetTheory.PGame.toLeftMovesAdd (SetTheory.PGame.toLeftMovesAdd.symm k))
  -/
  cases' toLeftMovesAdd.symm k with i i
    /-
      case inl
      x y : SetTheory.PGame
      k : (HAdd.hAdd x y).LeftMoves
      P : (HAdd.hAdd x y).LeftMoves → Prop
      hl : ∀ (i : x.LeftMoves), P (SetTheory.PGame.toLeftMovesAdd (Sum.inl i))
      hr : ∀ (i : y.LeftMoves), P (SetTheory.PGame.toLeftMovesAdd (Sum.inr i))
      i : x.LeftMoves
      ⊢ P (SetTheory.PGame.toLeftMovesAdd (Sum.inl i))
    -/
  · exact hl i
    /-
      🎉 no goals
    -/
    /-
      case inr
      x y : SetTheory.PGame
      k : (HAdd.hAdd x y).LeftMoves
      P : (HAdd.hAdd x y).LeftMoves → Prop
      hl : ∀ (i : x.LeftMoves), P (SetTheory.PGame.toLeftMovesAdd (Sum.inl i))
      hr : ∀ (i : y.LeftMoves), P (SetTheory.PGame.toLeftMovesAdd (Sum.inr i))
      i : y.LeftMoves
      ⊢ P (SetTheory.PGame.toLeftMovesAdd (Sum.inr i))
    -/
  · exact hr i
    /-
      🎉 no goals
    -/


/-- Case on possible right moves of `x + y`. -/
theorem rightMoves_add_cases {x y : PGame} (k) {P : (x + y).RightMoves → Prop}
    (hl : ∀ j, P <| toRightMovesAdd (Sum.inl j)) (hr : ∀ j, P <| toRightMovesAdd (Sum.inr j)) :
    P k := by
  /-
    x y : SetTheory.PGame
    k : (HAdd.hAdd x y).RightMoves
    P : (HAdd.hAdd x y).RightMoves → Prop
    hl : ∀ (j : x.RightMoves), P (SetTheory.PGame.toRightMovesAdd (Sum.inl j))
    hr : ∀ (j : y.RightMoves), P (SetTheory.PGame.toRightMovesAdd (Sum.inr j))
    ⊢ P k
  -/
  rw [← toRightMovesAdd.apply_symm_apply k]
  /-
    x y : SetTheory.PGame
    k : (HAdd.hAdd x y).RightMoves
    P : (HAdd.hAdd x y).RightMoves → Prop
    hl : ∀ (j : x.RightMoves), P (SetTheory.PGame.toRightMovesAdd (Sum.inl j))
    hr : ∀ (j : y.RightMoves), P (SetTheory.PGame.toRightMovesAdd (Sum.inr j))
    ⊢ P (SetTheory.PGame.toRightMovesAdd (SetTheory.PGame.toRightMovesAdd.symm k))
  -/
  cases' toRightMovesAdd.symm k with i i
    /-
      case inl
      x y : SetTheory.PGame
      k : (HAdd.hAdd x y).RightMoves
      P : (HAdd.hAdd x y).RightMoves → Prop
      hl : ∀ (j : x.RightMoves), P (SetTheory.PGame.toRightMovesAdd (Sum.inl j))
      hr : ∀ (j : y.RightMoves), P (SetTheory.PGame.toRightMovesAdd (Sum.inr j))
      i : x.RightMoves
      ⊢ P (SetTheory.PGame.toRightMovesAdd (Sum.inl i))
    -/
  · exact hl i
    /-
      🎉 no goals
    -/
    /-
      case inr
      x y : SetTheory.PGame
      k : (HAdd.hAdd x y).RightMoves
      P : (HAdd.hAdd x y).RightMoves → Prop
      hl : ∀ (j : x.RightMoves), P (SetTheory.PGame.toRightMovesAdd (Sum.inl j))
      hr : ∀ (j : y.RightMoves), P (SetTheory.PGame.toRightMovesAdd (Sum.inr j))
      i : y.RightMoves
      ⊢ P (SetTheory.PGame.toRightMovesAdd (Sum.inr i))
    -/
  · exact hr i
    /-
      🎉 no goals
    -/


instance isEmpty_nat_rightMoves : ∀ n : ℕ, IsEmpty (RightMoves n)
  | 0 => inferInstanceAs (IsEmpty PEmpty)
  | n + 1 => by
    /-
      xl xr : Type u
      n : Nat
      ⊢ IsEmpty (↑(HAdd.hAdd n 1)).RightMoves
    -/
    haveI := isEmpty_nat_rightMoves n
    /-
      xl xr : Type u
      n : Nat
      this : IsEmpty (↑n).RightMoves
      ⊢ IsEmpty (↑(HAdd.hAdd n 1)).RightMoves
    -/
    rw [PGame.nat_succ, rightMoves_add]
    /-
      xl xr : Type u
      n : Nat
      this : IsEmpty (↑n).RightMoves
      ⊢ IsEmpty (Sum (↑n).RightMoves (SetTheory.PGame.RightMoves 1))
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- If `w` has the same moves as `x` and `y` has the same moves as `z`,
then `w + y` has the same moves as `x + z`. -/
def Relabelling.addCongr : ∀ {w x y z : PGame.{u}}, w ≡r x → y ≡r z → w + y ≡r x + z
  | ⟨wl, wr, wL, wR⟩, ⟨xl, xr, xL, xR⟩, ⟨yl, yr, yL, yR⟩, ⟨zl, zr, zL, zR⟩, ⟨L₁, R₁, hL₁, hR₁⟩,
    ⟨L₂, R₂, hL₂, hR₂⟩ => by
    /-
      xl✝ xr✝ wl wr : Type u
      wL : wl → SetTheory.PGame
      wR : wr → SetTheory.PGame
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      zl zr : Type u
      zL : zl → SetTheory.PGame
      zR : zr → SetTheory.PGame
      L₁ : _root_.Equiv (SetTheory.PGame.mk wl wr wL wR).LeftMoves (SetTheory.PGame. …
      R₁ : _root_.Equiv (SetTheory.PGame.mk wl wr wL wR).RightMoves (SetTheory.PGame …
      hL₁ : (i : (SetTheory.PGame.mk wl wr wL wR).LeftMoves) → ((SetTheory.PGame.mk  …
      hR₁ : (j : (SetTheory.PGame.mk wl wr wL wR).RightMoves) → ((SetTheory.PGame.mk …
      L₂ : _root_.Equiv (SetTheory.PGame.mk yl yr yL yR).LeftMoves (SetTheory.PGame. …
      R₂ : _root_.Equiv (SetTheory.PGame.mk yl yr yL yR).RightMoves (SetTheory.PGame …
      hL₂ : (i : (SetTheory.PGame.mk yl yr yL yR).LeftMoves) → ((SetTheory.PGame.mk  …
      hR₂ : (j : (SetTheory.PGame.mk yl yr yL yR).RightMoves) → ((SetTheory.PGame.mk …
      ⊢ (HAdd.hAdd (SetTheory.PGame.mk wl wr wL wR) (SetTheory.PGame.mk yl yr yL yR) …
    -/
    let Hwx : ⟨wl, wr, wL, wR⟩ ≡r ⟨xl, xr, xL, xR⟩ := ⟨L₁, R₁, hL₁, hR₁⟩
    /-
      xl✝ xr✝ wl wr : Type u
      wL : wl → SetTheory.PGame
      wR : wr → SetTheory.PGame
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      zl zr : Type u
      zL : zl → SetTheory.PGame
      zR : zr → SetTheory.PGame
      L₁ : _root_.Equiv (SetTheory.PGame.mk wl wr wL wR).LeftMoves (SetTheory.PGame. …
      R₁ : _root_.Equiv (SetTheory.PGame.mk wl wr wL wR).RightMoves (SetTheory.PGame …
      hL₁ : (i : (SetTheory.PGame.mk wl wr wL wR).LeftMoves) → ((SetTheory.PGame.mk  …
      hR₁ : (j : (SetTheory.PGame.mk wl wr wL wR).RightMoves) → ((SetTheory.PGame.mk …
      L₂ : _root_.Equiv (SetTheory.PGame.mk yl yr yL yR).LeftMoves (SetTheory.PGame. …
      R₂ : _root_.Equiv (SetTheory.PGame.mk yl yr yL yR).RightMoves (SetTheory.PGame …
      hL₂ : (i : (SetTheory.PGame.mk yl yr yL yR).LeftMoves) → ((SetTheory.PGame.mk  …
      hR₂ : (j : (SetTheory.PGame.mk yl yr yL yR).RightMoves) → ((SetTheory.PGame.mk …
      Hwx : (SetTheory.PGame.mk wl wr wL wR).Relabelling (SetTheory.PGame.mk xl xr x …
      ⊢ (HAdd.hAdd (SetTheory.PGame.mk wl wr wL wR) (SetTheory.PGame.mk yl yr yL yR) …
    -/
    let Hyz : ⟨yl, yr, yL, yR⟩ ≡r ⟨zl, zr, zL, zR⟩ := ⟨L₂, R₂, hL₂, hR₂⟩
    /-
      xl✝ xr✝ wl wr : Type u
      wL : wl → SetTheory.PGame
      wR : wr → SetTheory.PGame
      xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      zl zr : Type u
      zL : zl → SetTheory.PGame
      zR : zr → SetTheory.PGame
      L₁ : _root_.Equiv (SetTheory.PGame.mk wl wr wL wR).LeftMoves (SetTheory.PGame. …
      R₁ : _root_.Equiv (SetTheory.PGame.mk wl wr wL wR).RightMoves (SetTheory.PGame …
      hL₁ : (i : (SetTheory.PGame.mk wl wr wL wR).LeftMoves) → ((SetTheory.PGame.mk  …
      hR₁ : (j : (SetTheory.PGame.mk wl wr wL wR).RightMoves) → ((SetTheory.PGame.mk …
      L₂ : _root_.Equiv (SetTheory.PGame.mk yl yr yL yR).LeftMoves (SetTheory.PGame. …
      R₂ : _root_.Equiv (SetTheory.PGame.mk yl yr yL yR).RightMoves (SetTheory.PGame …
      hL₂ : (i : (SetTheory.PGame.mk yl yr yL yR).LeftMoves) → ((SetTheory.PGame.mk  …
      hR₂ : (j : (SetTheory.PGame.mk yl yr yL yR).RightMoves) → ((SetTheory.PGame.mk …
      Hwx : (SetTheory.PGame.mk wl wr wL wR).Relabelling (SetTheory.PGame.mk xl xr x …
      Hyz : (SetTheory.PGame.mk yl yr yL yR).Relabelling (SetTheory.PGame.mk zl zr z …
      ⊢ (HAdd.hAdd (SetTheory.PGame.mk wl wr wL wR) (SetTheory.PGame.mk yl yr yL yR) …
    -/
    refine ⟨Equiv.sumCongr L₁ L₂, Equiv.sumCongr R₁ R₂, ?_, ?_⟩ <;> rintro (i | j)
      /-
        case refine_1.inl
        xl✝ xr✝ wl wr : Type u
        wL : wl → SetTheory.PGame
        wR : wr → SetTheory.PGame
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        L₁ : _root_.Equiv (SetTheory.PGame.mk wl wr wL wR).LeftMoves (SetTheory.PGame. …
        R₁ : _root_.Equiv (SetTheory.PGame.mk wl wr wL wR).RightMoves (SetTheory.PGame …
        hL₁ : (i : (SetTheory.PGame.mk wl wr wL wR).LeftMoves) → ((SetTheory.PGame.mk  …
        hR₁ : (j : (SetTheory.PGame.mk wl wr wL wR).RightMoves) → ((SetTheory.PGame.mk …
        L₂ : _root_.Equiv (SetTheory.PGame.mk yl yr yL yR).LeftMoves (SetTheory.PGame. …
        R₂ : _root_.Equiv (SetTheory.PGame.mk yl yr yL yR).RightMoves (SetTheory.PGame …
        hL₂ : (i : (SetTheory.PGame.mk yl yr yL yR).LeftMoves) → ((SetTheory.PGame.mk  …
        hR₂ : (j : (SetTheory.PGame.mk yl yr yL yR).RightMoves) → ((SetTheory.PGame.mk …
        Hwx : (SetTheory.PGame.mk wl wr wL wR).Relabelling (SetTheory.PGame.mk xl xr x …
        Hyz : (SetTheory.PGame.mk yl yr yL yR).Relabelling (SetTheory.PGame.mk zl zr z …
        i : wl
        ⊢ ((HAdd.hAdd (SetTheory.PGame.mk wl wr wL wR) (SetTheory.PGame.mk yl yr yL yR …
      -/
    · exact (hL₁ i).addCongr Hyz
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr
        xl✝ xr✝ wl wr : Type u
        wL : wl → SetTheory.PGame
        wR : wr → SetTheory.PGame
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        L₁ : _root_.Equiv (SetTheory.PGame.mk wl wr wL wR).LeftMoves (SetTheory.PGame. …
        R₁ : _root_.Equiv (SetTheory.PGame.mk wl wr wL wR).RightMoves (SetTheory.PGame …
        hL₁ : (i : (SetTheory.PGame.mk wl wr wL wR).LeftMoves) → ((SetTheory.PGame.mk  …
        hR₁ : (j : (SetTheory.PGame.mk wl wr wL wR).RightMoves) → ((SetTheory.PGame.mk …
        L₂ : _root_.Equiv (SetTheory.PGame.mk yl yr yL yR).LeftMoves (SetTheory.PGame. …
        R₂ : _root_.Equiv (SetTheory.PGame.mk yl yr yL yR).RightMoves (SetTheory.PGame …
        hL₂ : (i : (SetTheory.PGame.mk yl yr yL yR).LeftMoves) → ((SetTheory.PGame.mk  …
        hR₂ : (j : (SetTheory.PGame.mk yl yr yL yR).RightMoves) → ((SetTheory.PGame.mk …
        Hwx : (SetTheory.PGame.mk wl wr wL wR).Relabelling (SetTheory.PGame.mk xl xr x …
        Hyz : (SetTheory.PGame.mk yl yr yL yR).Relabelling (SetTheory.PGame.mk zl zr z …
        j : yl
        ⊢ ((HAdd.hAdd (SetTheory.PGame.mk wl wr wL wR) (SetTheory.PGame.mk yl yr yL yR …
      -/
    · exact Hwx.addCongr (hL₂ j)
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inl
        xl✝ xr✝ wl wr : Type u
        wL : wl → SetTheory.PGame
        wR : wr → SetTheory.PGame
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        L₁ : _root_.Equiv (SetTheory.PGame.mk wl wr wL wR).LeftMoves (SetTheory.PGame. …
        R₁ : _root_.Equiv (SetTheory.PGame.mk wl wr wL wR).RightMoves (SetTheory.PGame …
        hL₁ : (i : (SetTheory.PGame.mk wl wr wL wR).LeftMoves) → ((SetTheory.PGame.mk  …
        hR₁ : (j : (SetTheory.PGame.mk wl wr wL wR).RightMoves) → ((SetTheory.PGame.mk …
        L₂ : _root_.Equiv (SetTheory.PGame.mk yl yr yL yR).LeftMoves (SetTheory.PGame. …
        R₂ : _root_.Equiv (SetTheory.PGame.mk yl yr yL yR).RightMoves (SetTheory.PGame …
        hL₂ : (i : (SetTheory.PGame.mk yl yr yL yR).LeftMoves) → ((SetTheory.PGame.mk  …
        hR₂ : (j : (SetTheory.PGame.mk yl yr yL yR).RightMoves) → ((SetTheory.PGame.mk …
        Hwx : (SetTheory.PGame.mk wl wr wL wR).Relabelling (SetTheory.PGame.mk xl xr x …
        Hyz : (SetTheory.PGame.mk yl yr yL yR).Relabelling (SetTheory.PGame.mk zl zr z …
        i : wr
        ⊢ ((HAdd.hAdd (SetTheory.PGame.mk wl wr wL wR) (SetTheory.PGame.mk yl yr yL yR …
      -/
    · exact (hR₁ i).addCongr Hyz
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr
        xl✝ xr✝ wl wr : Type u
        wL : wl → SetTheory.PGame
        wR : wr → SetTheory.PGame
        xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        L₁ : _root_.Equiv (SetTheory.PGame.mk wl wr wL wR).LeftMoves (SetTheory.PGame. …
        R₁ : _root_.Equiv (SetTheory.PGame.mk wl wr wL wR).RightMoves (SetTheory.PGame …
        hL₁ : (i : (SetTheory.PGame.mk wl wr wL wR).LeftMoves) → ((SetTheory.PGame.mk  …
        hR₁ : (j : (SetTheory.PGame.mk wl wr wL wR).RightMoves) → ((SetTheory.PGame.mk …
        L₂ : _root_.Equiv (SetTheory.PGame.mk yl yr yL yR).LeftMoves (SetTheory.PGame. …
        R₂ : _root_.Equiv (SetTheory.PGame.mk yl yr yL yR).RightMoves (SetTheory.PGame …
        hL₂ : (i : (SetTheory.PGame.mk yl yr yL yR).LeftMoves) → ((SetTheory.PGame.mk  …
        hR₂ : (j : (SetTheory.PGame.mk yl yr yL yR).RightMoves) → ((SetTheory.PGame.mk …
        Hwx : (SetTheory.PGame.mk wl wr wL wR).Relabelling (SetTheory.PGame.mk xl xr x …
        Hyz : (SetTheory.PGame.mk yl yr yL yR).Relabelling (SetTheory.PGame.mk zl zr z …
        j : yr
        ⊢ ((HAdd.hAdd (SetTheory.PGame.mk wl wr wL wR) (SetTheory.PGame.mk yl yr yL yR …
      -/
    · exact Hwx.addCongr (hR₂ j)
      /-
        🎉 no goals
      -/
termination_by _ x _ z => (x, z)


instance : Sub PGame :=
  ⟨fun x y => x + -y⟩


@[simp]
theorem sub_zero_eq_add_zero (x : PGame) : x - 0 = x + 0 :=
                         /-
                           x : SetTheory.PGame
                           ⊢ Eq (HAdd.hAdd x (-0)) (HAdd.hAdd x 0)
                         -/
  show x + -0 = x + 0 by rw [neg_zero]
                         /-
                           🎉 no goals
                         -/


@[deprecated (since := "2024-09-26")] alias sub_zero := sub_zero_eq_add_zero


/-- If `w` has the same moves as `x` and `y` has the same moves as `z`,
then `w - y` has the same moves as `x - z`. -/
def Relabelling.subCongr {w x y z : PGame} (h₁ : w ≡r x) (h₂ : y ≡r z) : w - y ≡r x - z :=
  h₁.addCongr h₂.negCongr


/-- `-(x + y)` has exactly the same moves as `-x + -y`. -/
def negAddRelabelling : ∀ x y : PGame, -(x + y) ≡r -x + -y
  | ⟨xl, xr, xL, xR⟩, ⟨yl, yr, yL, yR⟩ => by
    /-
      xl✝ xr✝ : Type u
      xl xr : Type ?u.179248
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type ?u.179248
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      ⊢ (Neg.neg (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk yl  …
    -/
    refine ⟨Equiv.refl _, Equiv.refl _, ?_, ?_⟩
    all_goals
      exact fun j =>
        Sum.casesOn j (fun j => negAddRelabelling _ _) fun j =>
          negAddRelabelling ⟨xl, xr, xL, xR⟩ _
termination_by x y => (x, y)


theorem neg_add_le {x y : PGame} : -(x + y) ≤ -x + -y :=
  (negAddRelabelling x y).le


/-- `x + y` has exactly the same moves as `y + x`. -/
def addCommRelabelling : ∀ x y : PGame.{u}, x + y ≡r y + x
  | mk xl xr xL xR, mk yl yr yL yR => by
    /-
      xl✝ xr✝ xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      ⊢ (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk yl yr yL yR) …
    -/
    refine ⟨Equiv.sumComm _ _, Equiv.sumComm _ _, ?_, ?_⟩ <;> rintro (_ | _) <;>
        /-
          case refine_1.inl
          xl✝ xr✝ xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          val✝ : xl
          ⊢ ((HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk yl yr yL yR …
        -/
        /-
          case refine_1.inl
          xl✝ xr✝ xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          val✝ : xl
          ⊢ (HAdd.hAdd (xL val✝) (SetTheory.PGame.mk yl yr yL yR)).Relabelling ((HAdd.hA …
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
        /-
          case refine_2.inr
          xl✝ xr✝ xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          val✝ : yr
          ⊢ (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (yR val✝)).Relabelling ((HAdd.hA …
        -/
        apply addCommRelabelling
        /-
          🎉 no goals
        -/
termination_by x y => (x, y)


theorem add_comm_le {x y : PGame} : x + y ≤ y + x :=
  (addCommRelabelling x y).le


theorem add_comm_equiv {x y : PGame} : x + y ≈ y + x :=
  (addCommRelabelling x y).equiv


/-- `(x + y) + z` has exactly the same moves as `x + (y + z)`. -/
def addAssocRelabelling : ∀ x y z : PGame.{u}, x + y + z ≡r x + (y + z)
  | ⟨xl, xr, xL, xR⟩, ⟨yl, yr, yL, yR⟩, ⟨zl, zr, zL, zR⟩ => by
    /-
      xl✝ xr✝ xl xr : Type u
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      zl zr : Type u
      zL : zl → SetTheory.PGame
      zR : zr → SetTheory.PGame
      ⊢ (HAdd.hAdd (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk y …
    -/
    refine ⟨Equiv.sumAssoc _ _ _, Equiv.sumAssoc _ _ _, ?_, ?_⟩
      /-
        case refine_1
        xl✝ xr✝ xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        ⊢ (i : (HAdd.hAdd (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame …
      -/
    · rintro (⟨i | i⟩ | i)
        /-
          case refine_1.inl.inl
          xl✝ xr✝ xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          i : xl
          ⊢ ((HAdd.hAdd (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk  …
        -/
      · apply addAssocRelabelling
        /-
          🎉 no goals
        -/
        /-
          case refine_1.inl.inr
          xl✝ xr✝ xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          i : yl
          ⊢ ((HAdd.hAdd (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk  …
        -/
      · apply addAssocRelabelling ⟨xl, xr, xL, xR⟩ (yL i)
        /-
          🎉 no goals
        -/
        /-
          case refine_1.inr
          xl✝ xr✝ xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          i : zl
          ⊢ ((HAdd.hAdd (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk  …
        -/
      · apply addAssocRelabelling ⟨xl, xr, xL, xR⟩ ⟨yl, yr, yL, yR⟩ (zL i)
        /-
          🎉 no goals
        -/
      /-
        case refine_2
        xl✝ xr✝ xl xr : Type u
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        ⊢ (j : (HAdd.hAdd (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame …
      -/
    · rintro (⟨i | i⟩ | i)
        /-
          case refine_2.inl.inl
          xl✝ xr✝ xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          i : xr
          ⊢ ((HAdd.hAdd (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk  …
        -/
      · apply addAssocRelabelling
        /-
          🎉 no goals
        -/
        /-
          case refine_2.inl.inr
          xl✝ xr✝ xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          i : yr
          ⊢ ((HAdd.hAdd (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk  …
        -/
      · apply addAssocRelabelling ⟨xl, xr, xL, xR⟩ (yR i)
        /-
          🎉 no goals
        -/
        /-
          case refine_2.inr
          xl✝ xr✝ xl xr : Type u
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          i : zr
          ⊢ ((HAdd.hAdd (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk  …
        -/
      · apply addAssocRelabelling ⟨xl, xr, xL, xR⟩ ⟨yl, yr, yL, yR⟩ (zR i)
        /-
          🎉 no goals
        -/
termination_by x y z => (x, y, z)


theorem add_assoc_equiv {x y z : PGame} : x + y + z ≈ x + (y + z) :=
  (addAssocRelabelling x y z).equiv


theorem neg_add_cancel_le_zero : ∀ x : PGame, -x + x ≤ 0
  | ⟨xl, xr, xL, xR⟩ =>
    le_zero.2 fun i => by
      /-
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        i : (HAdd.hAdd (Neg.neg (SetTheory.PGame.mk xl xr xL xR)) (SetTheory.PGame.mk  …
        ⊢ Exists fun j => LE.le (((HAdd.hAdd (Neg.neg (SetTheory.PGame.mk xl xr xL xR) …
      -/
      cases' i with i i
      · -- If Left played in -x, Right responds with the same move in x.
        /-
          case inl
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          i : xr
          ⊢ Exists fun j => LE.le (((HAdd.hAdd (Neg.neg (SetTheory.PGame.mk xl xr xL xR) …
        -/
        refine ⟨@toRightMovesAdd _ ⟨_, _, _, _⟩ (Sum.inr i), ?_⟩
        /-
          case inl
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          i : xr
          ⊢ LE.le (((HAdd.hAdd (Neg.neg (SetTheory.PGame.mk xl xr xL xR)) (SetTheory.PGa …
        -/
        convert @neg_add_cancel_le_zero (xR i)
        /-
          case h.e'_3
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          i : xr
          ⊢ Eq (((HAdd.hAdd (Neg.neg (SetTheory.PGame.mk xl xr xL xR)) (SetTheory.PGame. …
        -/
        apply add_moveRight_inr
        /-
          🎉 no goals
        -/
      · -- If Left in x, Right responds with the same move in -x.
        /-
          case inr
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          i : xl
          ⊢ Exists fun j => LE.le (((HAdd.hAdd (Neg.neg (SetTheory.PGame.mk xl xr xL xR) …
        -/
        dsimp
        /-
          case inr
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          i : xl
          ⊢ Exists fun j => LE.le ((HAdd.hAdd (SetTheory.PGame.mk xr xl (fun j => Neg.ne …
        -/
        refine ⟨@toRightMovesAdd ⟨_, _, _, _⟩ _ (Sum.inl i), ?_⟩
        /-
          case inr
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          i : xl
          ⊢ LE.le ((HAdd.hAdd (SetTheory.PGame.mk xr xl (fun j => Neg.neg (xR j)) fun i  …
        -/
        convert @neg_add_cancel_le_zero (xL i)
        /-
          case h.e'_3
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          i : xl
          ⊢ Eq ((HAdd.hAdd (SetTheory.PGame.mk xr xl (fun j => Neg.neg (xR j)) fun i =>  …
        -/
        apply add_moveRight_inl
        /-
          🎉 no goals
        -/


theorem zero_le_neg_add_cancel (x : PGame) : 0 ≤ -x + x := by
  /-
    x : SetTheory.PGame
    ⊢ LE.le 0 (HAdd.hAdd (Neg.neg x) x)
  -/
  rw [← neg_le_neg_iff, neg_zero]
  /-
    x : SetTheory.PGame
    ⊢ LE.le (Neg.neg (HAdd.hAdd (Neg.neg x) x)) 0
  -/
  exact neg_add_le.trans (neg_add_cancel_le_zero _)
  /-
    🎉 no goals
  -/


theorem neg_add_cancel_equiv (x : PGame) : -x + x ≈ 0 :=
  ⟨neg_add_cancel_le_zero x, zero_le_neg_add_cancel x⟩


theorem add_neg_cancel_le_zero (x : PGame) : x + -x ≤ 0 :=
  add_comm_le.trans (neg_add_cancel_le_zero x)


theorem zero_le_add_neg_cancel (x : PGame) : 0 ≤ x + -x :=
  (zero_le_neg_add_cancel x).trans add_comm_le


theorem add_neg_cancel_equiv (x : PGame) : x + -x ≈ 0 :=
  ⟨add_neg_cancel_le_zero x, zero_le_add_neg_cancel x⟩


theorem sub_self_equiv : ∀ (x : PGame), x - x ≈ 0 :=
  add_neg_cancel_equiv


private theorem add_le_add_right' : ∀ {x y z : PGame}, x ≤ y → x + z ≤ y + z
  | mk xl xr xL xR, mk yl yr yL yR, mk zl zr zL zR => fun h => by
    /-
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      zl zr : Type u_1
      zL : zl → SetTheory.PGame
      zR : zr → SetTheory.PGame
      h : LE.le (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk yl yr yL yR)
      ⊢ LE.le (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk zl zr  …
    -/
    refine le_def.2 ⟨fun i => ?_, fun i => ?_⟩ <;> cases' i with i i
      /-
        case refine_1.inl
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u_1
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        h : LE.le (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk yl yr yL yR)
        i : xl
        ⊢ Or (Exists fun i' => LE.le ((HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (Set …
      -/
    · rw [le_def] at h
      /-
        case refine_1.inl
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u_1
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        h : And (∀ (i : (SetTheory.PGame.mk xl xr xL xR).LeftMoves), Or (Exists fun i' …
        i : xl
        ⊢ Or (Exists fun i' => LE.le ((HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (Set …
      -/
      cases' h with h_left h_right
      /-
        case refine_1.inl.intro
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u_1
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        i : xl
        h_left : ∀ (i : (SetTheory.PGame.mk xl xr xL xR).LeftMoves), Or (Exists fun i' …
        h_right : ∀ (j : (SetTheory.PGame.mk yl yr yL yR).RightMoves), Or (Exists fun  …
        ⊢ Or (Exists fun i' => LE.le ((HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (Set …
      -/
      rcases h_left i with (⟨i', ih⟩ | ⟨j, jh⟩)
        /-
          case refine_1.inl.intro.inl.intro
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          i : xl
          h_left : ∀ (i : (SetTheory.PGame.mk xl xr xL xR).LeftMoves), Or (Exists fun i' …
          h_right : ∀ (j : (SetTheory.PGame.mk yl yr yL yR).RightMoves), Or (Exists fun  …
          i' : (SetTheory.PGame.mk yl yr yL yR).LeftMoves
          ih : LE.le ((SetTheory.PGame.mk xl xr xL xR).moveLeft i) ((SetTheory.PGame.mk  …
          ⊢ Or (Exists fun i' => LE.le ((HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (Set …
        -/
      · exact Or.inl ⟨toLeftMovesAdd (Sum.inl i'), add_le_add_right' ih⟩
        /-
          🎉 no goals
        -/
        /-
          case refine_1.inl.intro.inr.intro
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          i : xl
          h_left : ∀ (i : (SetTheory.PGame.mk xl xr xL xR).LeftMoves), Or (Exists fun i' …
          h_right : ∀ (j : (SetTheory.PGame.mk yl yr yL yR).RightMoves), Or (Exists fun  …
          j : ((SetTheory.PGame.mk xl xr xL xR).moveLeft i).RightMoves
          jh : LE.le (((SetTheory.PGame.mk xl xr xL xR).moveLeft i).moveRight j) (SetThe …
          ⊢ Or (Exists fun i' => LE.le ((HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (Set …
        -/
      · refine Or.inr ⟨toRightMovesAdd (Sum.inl j), ?_⟩
        /-
          case refine_1.inl.intro.inr.intro
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          i : xl
          h_left : ∀ (i : (SetTheory.PGame.mk xl xr xL xR).LeftMoves), Or (Exists fun i' …
          h_right : ∀ (j : (SetTheory.PGame.mk yl yr yL yR).RightMoves), Or (Exists fun  …
          j : ((SetTheory.PGame.mk xl xr xL xR).moveLeft i).RightMoves
          jh : LE.le (((SetTheory.PGame.mk xl xr xL xR).moveLeft i).moveRight j) (SetThe …
          ⊢ LE.le (((HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk zl z …
        -/
        convert add_le_add_right' jh
        /-
          case h.e'_3
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          i : xl
          h_left : ∀ (i : (SetTheory.PGame.mk xl xr xL xR).LeftMoves), Or (Exists fun i' …
          h_right : ∀ (j : (SetTheory.PGame.mk yl yr yL yR).RightMoves), Or (Exists fun  …
          j : ((SetTheory.PGame.mk xl xr xL xR).moveLeft i).RightMoves
          jh : LE.le (((SetTheory.PGame.mk xl xr xL xR).moveLeft i).moveRight j) (SetThe …
          ⊢ Eq (((HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk zl zr z …
        -/
        apply add_moveRight_inl
        /-
          🎉 no goals
        -/
      /-
        case refine_1.inr
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u_1
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        h : LE.le (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk yl yr yL yR)
        i : zl
        ⊢ Or (Exists fun i' => LE.le ((HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (Set …
      -/
    · exact Or.inl ⟨@toLeftMovesAdd _ ⟨_, _, _, _⟩ (Sum.inr i), add_le_add_right' h⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inl
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u_1
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        h : LE.le (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk yl yr yL yR)
        i : yr
        ⊢ Or (Exists fun i_1 => LE.le (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (Set …
      -/
    · rw [le_def] at h
      /-
        case refine_2.inl
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        zl zr : Type u_1
        zL : zl → SetTheory.PGame
        zR : zr → SetTheory.PGame
        h : And (∀ (i : (SetTheory.PGame.mk xl xr xL xR).LeftMoves), Or (Exists fun i' …
        i : yr
        ⊢ Or (Exists fun i_1 => LE.le (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (Set …
      -/
      rcases h.right i with (⟨i, ih⟩ | ⟨j', jh⟩)
        /-
          case refine_2.inl.inl.intro
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          h : And (∀ (i : (SetTheory.PGame.mk xl xr xL xR).LeftMoves), Or (Exists fun i' …
          i✝ : yr
          i : ((SetTheory.PGame.mk yl yr yL yR).moveRight i✝).LeftMoves
          ih : LE.le (SetTheory.PGame.mk xl xr xL xR) (((SetTheory.PGame.mk yl yr yL yR) …
          ⊢ Or (Exists fun i => LE.le (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTh …
        -/
      · refine Or.inl ⟨toLeftMovesAdd (Sum.inl i), ?_⟩
        /-
          case refine_2.inl.inl.intro
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          h : And (∀ (i : (SetTheory.PGame.mk xl xr xL xR).LeftMoves), Or (Exists fun i' …
          i✝ : yr
          i : ((SetTheory.PGame.mk yl yr yL yR).moveRight i✝).LeftMoves
          ih : LE.le (SetTheory.PGame.mk xl xr xL xR) (((SetTheory.PGame.mk yl yr yL yR) …
          ⊢ LE.le (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (SetTheory.PGame.mk zl zr  …
        -/
        convert add_le_add_right' ih
        /-
          case h.e'_4
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          h : And (∀ (i : (SetTheory.PGame.mk xl xr xL xR).LeftMoves), Or (Exists fun i' …
          i✝ : yr
          i : ((SetTheory.PGame.mk yl yr yL yR).moveRight i✝).LeftMoves
          ih : LE.le (SetTheory.PGame.mk xl xr xL xR) (((SetTheory.PGame.mk yl yr yL yR) …
          ⊢ Eq (((HAdd.hAdd (SetTheory.PGame.mk yl yr yL yR) (SetTheory.PGame.mk zl zr z …
        -/
        apply add_moveLeft_inl
        /-
          🎉 no goals
        -/
        /-
          case refine_2.inl.inr.intro
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          zl zr : Type u_1
          zL : zl → SetTheory.PGame
          zR : zr → SetTheory.PGame
          h : And (∀ (i : (SetTheory.PGame.mk xl xr xL xR).LeftMoves), Or (Exists fun i' …
          i : yr
          j' : (SetTheory.PGame.mk xl xr xL xR).RightMoves
          jh : LE.le ((SetTheory.PGame.mk xl xr xL xR).moveRight j') ((SetTheory.PGame.m …
          ⊢ Or (Exists fun i_1 => LE.le (HAdd.hAdd (SetTheory.PGame.mk xl xr xL xR) (Set …
        -/
      · exact Or.inr ⟨toRightMovesAdd (Sum.inl j'), add_le_add_right' jh⟩
        /-
          🎉 no goals
        -/
    · exact
        Or.inr ⟨@toRightMovesAdd _ ⟨_, _, _, _⟩ (Sum.inr i), add_le_add_right' h⟩
termination_by x y z => (x, y, z)


instance addRightMono : AddRightMono PGame :=
  ⟨fun _ _ _ => add_le_add_right'⟩


instance addLeftMono : AddLeftMono PGame :=
  ⟨fun x _ _ h => (add_comm_le.trans (add_le_add_right h x)).trans add_comm_le⟩


theorem add_lf_add_right {y z : PGame} (h : y ⧏ z) (x) : y + x ⧏ z + x :=
  suffices z + x ≤ y + x → z ≤ y by
    /-
      y z : SetTheory.PGame
      h : y.LF z
      x : SetTheory.PGame
      this : LE.le (HAdd.hAdd z x) (HAdd.hAdd y x) → LE.le z y
      ⊢ (HAdd.hAdd y x).LF (HAdd.hAdd z x)
    -/
    rw [← PGame.not_le] at h ⊢
    /-
      y z : SetTheory.PGame
      h : Not (LE.le z y)
      x : SetTheory.PGame
      this : LE.le (HAdd.hAdd z x) (HAdd.hAdd y x) → LE.le z y
      ⊢ Not (LE.le (HAdd.hAdd z x) (HAdd.hAdd y x))
    -/
    exact mt this h
    /-
      🎉 no goals
    -/
  fun w =>
  calc
    z ≤ z + 0 := (addZeroRelabelling _).symm.le
    _ ≤ z + (x + -x) := add_le_add_left (zero_le_add_neg_cancel x) _
    _ ≤ z + x + -x := (addAssocRelabelling _ _ _).symm.le
    _ ≤ y + x + -x := add_le_add_right w _
    _ ≤ y + (x + -x) := (addAssocRelabelling _ _ _).le
    _ ≤ y + 0 := add_le_add_left (add_neg_cancel_le_zero x) _
    _ ≤ y := (addZeroRelabelling _).le


theorem add_lf_add_left {y z : PGame} (h : y ⧏ z) (x) : x + y ⧏ x + z := by
  /-
    y z : SetTheory.PGame
    h : y.LF z
    x : SetTheory.PGame
    ⊢ (HAdd.hAdd x y).LF (HAdd.hAdd x z)
  -/
  rw [lf_congr add_comm_equiv add_comm_equiv]
  /-
    y z : SetTheory.PGame
    h : y.LF z
    x : SetTheory.PGame
    ⊢ (HAdd.hAdd y x).LF (HAdd.hAdd z x)
  -/
  apply add_lf_add_right h
  /-
    🎉 no goals
  -/


instance addRightStrictMono : AddRightStrictMono PGame :=
  ⟨fun x _ _ h => ⟨add_le_add_right h.1 x, add_lf_add_right h.2 x⟩⟩


instance addLeftStrictMono : AddLeftStrictMono PGame :=
  ⟨fun x _ _ h => ⟨add_le_add_left h.1 x, add_lf_add_left h.2 x⟩⟩


theorem add_lf_add_of_lf_of_le {w x y z : PGame} (hwx : w ⧏ x) (hyz : y ≤ z) : w + y ⧏ x + z :=
  lf_of_lf_of_le (add_lf_add_right hwx y) (add_le_add_left hyz x)


theorem add_lf_add_of_le_of_lf {w x y z : PGame} (hwx : w ≤ x) (hyz : y ⧏ z) : w + y ⧏ x + z :=
  lf_of_le_of_lf (add_le_add_right hwx y) (add_lf_add_left hyz x)


theorem add_congr {w x y z : PGame} (h₁ : w ≈ x) (h₂ : y ≈ z) : w + y ≈ x + z :=
  ⟨(add_le_add_left h₂.1 w).trans (add_le_add_right h₁.1 z),
    (add_le_add_left h₂.2 x).trans (add_le_add_right h₁.2 y)⟩


theorem add_congr_left {x y z : PGame} (h : x ≈ y) : x + z ≈ y + z :=
  add_congr h equiv_rfl


theorem add_congr_right {x y z : PGame} : (y ≈ z) → (x + y ≈ x + z) :=
  add_congr equiv_rfl


theorem sub_congr {w x y z : PGame} (h₁ : w ≈ x) (h₂ : y ≈ z) : w - y ≈ x - z :=
  add_congr h₁ (neg_equiv_neg_iff.2 h₂)


theorem sub_congr_left {x y z : PGame} (h : x ≈ y) : x - z ≈ y - z :=
  sub_congr h equiv_rfl


theorem sub_congr_right {x y z : PGame} : (y ≈ z) → (x - y ≈ x - z) :=
  sub_congr equiv_rfl


theorem le_iff_sub_nonneg {x y : PGame} : x ≤ y ↔ 0 ≤ y - x :=
  ⟨fun h => (zero_le_add_neg_cancel x).trans (add_le_add_right h _), fun h =>
    calc
      x ≤ 0 + x := (zeroAddRelabelling x).symm.le
      _ ≤ y - x + x := add_le_add_right h _
      _ ≤ y + (-x + x) := (addAssocRelabelling _ _ _).le
      _ ≤ y + 0 := add_le_add_left (neg_add_cancel_le_zero x) _
      _ ≤ y := (addZeroRelabelling y).le
      ⟩


theorem lf_iff_sub_zero_lf {x y : PGame} : x ⧏ y ↔ 0 ⧏ y - x :=
  ⟨fun h => (zero_le_add_neg_cancel x).trans_lf (add_lf_add_right h _), fun h =>
    calc
      x ≤ 0 + x := (zeroAddRelabelling x).symm.le
      _ ⧏ y - x + x := add_lf_add_right h _
      _ ≤ y + (-x + x) := (addAssocRelabelling _ _ _).le
      _ ≤ y + 0 := add_le_add_left (neg_add_cancel_le_zero x) _
      _ ≤ y := (addZeroRelabelling y).le
      ⟩


theorem lt_iff_sub_pos {x y : PGame} : x < y ↔ 0 < y - x :=
  ⟨fun h => lt_of_le_of_lt (zero_le_add_neg_cancel x) (add_lt_add_right h _), fun h =>
    calc
      x ≤ 0 + x := (zeroAddRelabelling x).symm.le
      _ < y - x + x := add_lt_add_right h _
      _ ≤ y + (-x + x) := (addAssocRelabelling _ _ _).le
      _ ≤ y + 0 := add_le_add_left (neg_add_cancel_le_zero x) _
      _ ≤ y := (addZeroRelabelling y).le
      ⟩


/-- The pregame constructed by inserting `x'` as a new left option into x. -/
def insertLeft (x x' : PGame.{u}) : PGame :=
  match x with
  | mk xl xr xL xR => mk (xl ⊕ PUnit) xr (Sum.elim xL fun _ => x') xR


/-- A new left option cannot hurt Left. -/
lemma le_insertLeft (x x' : PGame) : x ≤ insertLeft x x' := by
  /-
    x x' : SetTheory.PGame
    ⊢ LE.le x (x.insertLeft x')
  -/
  rw [le_def]
  /-
    x x' : SetTheory.PGame
    ⊢ And (∀ (i : x.LeftMoves), Or (Exists fun i' => LE.le (x.moveLeft i) ((x.inse …
  -/
  constructor
    /-
      case left
      x x' : SetTheory.PGame
      ⊢ ∀ (i : x.LeftMoves), Or (Exists fun i' => LE.le (x.moveLeft i) ((x.insertLef …
    -/
  · intro i
    /-
      case left
      x x' : SetTheory.PGame
      i : x.LeftMoves
      ⊢ Or (Exists fun i' => LE.le (x.moveLeft i) ((x.insertLeft x').moveLeft i')) ( …
    -/
    left
    /-
      case left.h
      x x' : SetTheory.PGame
      i : x.LeftMoves
      ⊢ Exists fun i' => LE.le (x.moveLeft i) ((x.insertLeft x').moveLeft i')
    -/
    rcases x with ⟨xl, xr, xL, xR⟩
    /-
      case left.h.mk
      x' : SetTheory.PGame
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      i : (SetTheory.PGame.mk xl xr xL xR).LeftMoves
      ⊢ Exists fun i' => LE.le ((SetTheory.PGame.mk xl xr xL xR).moveLeft i) (((SetT …
    -/
    simp only [insertLeft, leftMoves_mk, moveLeft_mk, Sum.exists, Sum.elim_inl]
    /-
      case left.h.mk
      x' : SetTheory.PGame
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      i : (SetTheory.PGame.mk xl xr xL xR).LeftMoves
      ⊢ Or (Exists fun a => LE.le (xL i) (xL a)) (Exists fun b => LE.le (xL i) (Sum. …
    -/
    left
    /-
      case left.h.mk.h
      x' : SetTheory.PGame
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      i : (SetTheory.PGame.mk xl xr xL xR).LeftMoves
      ⊢ Exists fun a => LE.le (xL i) (xL a)
    -/
    use i
    /-
      🎉 no goals
    -/
    /-
      case right
      x x' : SetTheory.PGame
      ⊢ ∀ (j : (x.insertLeft x').RightMoves), Or (Exists fun i => LE.le x (((x.inser …
    -/
  · intro j
    /-
      case right
      x x' : SetTheory.PGame
      j : (x.insertLeft x').RightMoves
      ⊢ Or (Exists fun i => LE.le x (((x.insertLeft x').moveRight j).moveLeft i)) (E …
    -/
    right
    /-
      case right.h
      x x' : SetTheory.PGame
      j : (x.insertLeft x').RightMoves
      ⊢ Exists fun j' => LE.le (x.moveRight j') ((x.insertLeft x').moveRight j)
    -/
    rcases x with ⟨xl, xr, xL, xR⟩
    /-
      case right.h.mk
      x' : SetTheory.PGame
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      j : ((SetTheory.PGame.mk xl xr xL xR).insertLeft x').RightMoves
      ⊢ Exists fun j' => LE.le ((SetTheory.PGame.mk xl xr xL xR).moveRight j') (((Se …
    -/
    simp only [rightMoves_mk, moveRight_mk, insertLeft]
    /-
      case right.h.mk
      x' : SetTheory.PGame
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      j : ((SetTheory.PGame.mk xl xr xL xR).insertLeft x').RightMoves
      ⊢ Exists fun j' => LE.le (xR j') (xR j)
    -/
    use j
    /-
      🎉 no goals
    -/


/-- Adding a gift horse left option does not change the value of `x`. A gift horse left option is
 a game `x'` with `x' ⧏ x`. It is called "gift horse" because it seems like Left has gotten the
 "gift" of a new option, but actually the value of the game did not change. -/
lemma insertLeft_equiv_of_lf {x x' : PGame} (h : x' ⧏ x) : insertLeft x x' ≈ x := by
  /-
    x x' : SetTheory.PGame
    h : x'.LF x
    ⊢ HasEquiv.Equiv (x.insertLeft x') x
  -/
  rw [equiv_def]
  /-
    x x' : SetTheory.PGame
    h : x'.LF x
    ⊢ And (LE.le (x.insertLeft x') x) (LE.le x (x.insertLeft x'))
  -/
  constructor
    /-
      case left
      x x' : SetTheory.PGame
      h : x'.LF x
      ⊢ LE.le (x.insertLeft x') x
    -/
  · rw [le_def]
    /-
      case left
      x x' : SetTheory.PGame
      h : x'.LF x
      ⊢ And (∀ (i : (x.insertLeft x').LeftMoves), Or (Exists fun i' => LE.le ((x.ins …
    -/
    constructor
      /-
        case left.left
        x x' : SetTheory.PGame
        h : x'.LF x
        ⊢ ∀ (i : (x.insertLeft x').LeftMoves), Or (Exists fun i' => LE.le ((x.insertLe …
      -/
    · intro i
      /-
        case left.left
        x x' : SetTheory.PGame
        h : x'.LF x
        i : (x.insertLeft x').LeftMoves
        ⊢ Or (Exists fun i' => LE.le ((x.insertLeft x').moveLeft i) (x.moveLeft i')) ( …
      -/
      rcases x with ⟨xl, xr, xL, xR⟩
      /-
        case left.left.mk
        x' : SetTheory.PGame
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        h : x'.LF (SetTheory.PGame.mk xl xr xL xR)
        i : ((SetTheory.PGame.mk xl xr xL xR).insertLeft x').LeftMoves
        ⊢ Or (Exists fun i' => LE.le (((SetTheory.PGame.mk xl xr xL xR).insertLeft x') …
      -/
      simp only [insertLeft, leftMoves_mk, moveLeft_mk] at i ⊢
      /-
        case left.left.mk
        x' : SetTheory.PGame
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        h : x'.LF (SetTheory.PGame.mk xl xr xL xR)
        i : Sum xl PUnit.{u_1 + 1}
        ⊢ Or (Exists fun i' => LE.le (Sum.elim xL (fun x => x') i) (xL i')) (Exists fu …
      -/
      rcases i with i | _
        /-
          case left.left.mk.inl
          x' : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          h : x'.LF (SetTheory.PGame.mk xl xr xL xR)
          i : xl
          ⊢ Or (Exists fun i' => LE.le (Sum.elim xL (fun x => x') (Sum.inl i)) (xL i'))  …
        -/
      · simp only [Sum.elim_inl]
        /-
          case left.left.mk.inl
          x' : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          h : x'.LF (SetTheory.PGame.mk xl xr xL xR)
          i : xl
          ⊢ Or (Exists fun i' => LE.le (xL i) (xL i')) (Exists fun j => LE.le ((xL i).mo …
        -/
        left
        /-
          case left.left.mk.inl.h
          x' : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          h : x'.LF (SetTheory.PGame.mk xl xr xL xR)
          i : xl
          ⊢ Exists fun i' => LE.le (xL i) (xL i')
        -/
        use i
        /-
          🎉 no goals
        -/
        /-
          case left.left.mk.inr
          x' : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          h : x'.LF (SetTheory.PGame.mk xl xr xL xR)
          val✝ : PUnit.{u_1 + 1}
          ⊢ Or (Exists fun i' => LE.le (Sum.elim xL (fun x => x') (Sum.inr val✝)) (xL i' …
        -/
      · simp only [Sum.elim_inr]
        /-
          case left.left.mk.inr
          x' : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          h : x'.LF (SetTheory.PGame.mk xl xr xL xR)
          val✝ : PUnit.{u_1 + 1}
          ⊢ Or (Exists fun i' => LE.le x' (xL i')) (Exists fun j => LE.le (x'.moveRight  …
        -/
        rw [lf_iff_exists_le] at h
        /-
          case left.left.mk.inr
          x' : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          h : Or (Exists fun i => LE.le x' ((SetTheory.PGame.mk xl xr xL xR).moveLeft i) …
          val✝ : PUnit.{u_1 + 1}
          ⊢ Or (Exists fun i' => LE.le x' (xL i')) (Exists fun j => LE.le (x'.moveRight  …
        -/
        simp only [leftMoves_mk, moveLeft_mk] at h
        /-
          case left.left.mk.inr
          x' : SetTheory.PGame
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          h : Or (Exists fun i => LE.le x' (xL i)) (Exists fun j => LE.le (x'.moveRight  …
          val✝ : PUnit.{u_1 + 1}
          ⊢ Or (Exists fun i' => LE.le x' (xL i')) (Exists fun j => LE.le (x'.moveRight  …
        -/
        exact h
        /-
          🎉 no goals
        -/
      /-
        case left.right
        x x' : SetTheory.PGame
        h : x'.LF x
        ⊢ ∀ (j : x.RightMoves), Or (Exists fun i => LE.le (x.insertLeft x') ((x.moveRi …
      -/
    · intro j
      /-
        case left.right
        x x' : SetTheory.PGame
        h : x'.LF x
        j : x.RightMoves
        ⊢ Or (Exists fun i => LE.le (x.insertLeft x') ((x.moveRight j).moveLeft i)) (E …
      -/
      right
      /-
        case left.right.h
        x x' : SetTheory.PGame
        h : x'.LF x
        j : x.RightMoves
        ⊢ Exists fun j' => LE.le ((x.insertLeft x').moveRight j') (x.moveRight j)
      -/
      rcases x with ⟨xl, xr, xL, xR⟩
      /-
        case left.right.h.mk
        x' : SetTheory.PGame
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        h : x'.LF (SetTheory.PGame.mk xl xr xL xR)
        j : (SetTheory.PGame.mk xl xr xL xR).RightMoves
        ⊢ Exists fun j' => LE.le (((SetTheory.PGame.mk xl xr xL xR).insertLeft x').mov …
      -/
      simp only [insertLeft, rightMoves_mk, moveRight_mk]
      /-
        case left.right.h.mk
        x' : SetTheory.PGame
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        h : x'.LF (SetTheory.PGame.mk xl xr xL xR)
        j : (SetTheory.PGame.mk xl xr xL xR).RightMoves
        ⊢ Exists fun j' => LE.le (xR j') (xR j)
      -/
      use j
      /-
        🎉 no goals
      -/
    /-
      case right
      x x' : SetTheory.PGame
      h : x'.LF x
      ⊢ LE.le x (x.insertLeft x')
    -/
  · apply le_insertLeft
    /-
      🎉 no goals
    -/


/-- The pregame constructed by inserting `x'` as a new right option into x. -/
def insertRight (x x' : PGame.{u}) : PGame :=
  match x with
  | mk xl xr xL xR => mk xl (xr ⊕ PUnit) xL (Sum.elim xR fun _ => x')


theorem neg_insertRight_neg (x x' : PGame.{u}) : (-x).insertRight (-x') = -x.insertLeft x' := by
  /-
    x x' : SetTheory.PGame
    ⊢ Eq ((Neg.neg x).insertRight (Neg.neg x')) (Neg.neg (x.insertLeft x'))
  -/
  cases x
  /-
    case mk
    x' : SetTheory.PGame
    α✝ β✝ : Type u
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    ⊢ Eq ((Neg.neg (SetTheory.PGame.mk α✝ β✝ a✝¹ a✝)).insertRight (Neg.neg x')) (N …
  -/
  cases x'
  /-
    case mk.mk
    α✝¹ β✝¹ : Type u
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    α✝ β✝ : Type u
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    ⊢ Eq ((Neg.neg (SetTheory.PGame.mk α✝¹ β✝¹ a✝³ a✝²)).insertRight (Neg.neg (Set …
  -/
  dsimp [insertRight, insertLeft]
  /-
    case mk.mk
    α✝¹ β✝¹ : Type u
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    α✝ β✝ : Type u
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    ⊢ Eq (SetTheory.PGame.mk β✝¹ (Sum α✝¹ PUnit.{u + 1}) (fun j => Neg.neg (a✝² j) …
  -/
  congr! with (i | j)
  /-
    🎉 no goals
  -/


theorem neg_insertLeft_neg (x x' : PGame.{u}) : (-x).insertLeft (-x') = -x.insertRight x' := by
  /-
    x x' : SetTheory.PGame
    ⊢ Eq ((Neg.neg x).insertLeft (Neg.neg x')) (Neg.neg (x.insertRight x'))
  -/
  rw [← neg_eq_iff_eq_neg, ← neg_insertRight_neg, neg_neg, neg_neg]
  /-
    🎉 no goals
  -/


/-- A new right option cannot hurt Right. -/
lemma insertRight_le (x x' : PGame) : insertRight x x' ≤ x := by
  /-
    x x' : SetTheory.PGame
    ⊢ LE.le (x.insertRight x') x
  -/
  rw [← neg_le_neg_iff, ← neg_insertLeft_neg]
  /-
    x x' : SetTheory.PGame
    ⊢ LE.le (Neg.neg x) ((Neg.neg x).insertLeft (Neg.neg x'))
  -/
  exact le_insertLeft _ _
  /-
    🎉 no goals
  -/


/-- Adding a gift horse right option does not change the value of `x`. A gift horse right option is
 a game `x'` with `x ⧏ x'`. It is called "gift horse" because it seems like Right has gotten the
 "gift" of a new option, but actually the value of the game did not change. -/
lemma insertRight_equiv_of_lf {x x' : PGame} (h : x ⧏ x') : insertRight x x' ≈ x := by
  /-
    x x' : SetTheory.PGame
    h : x.LF x'
    ⊢ HasEquiv.Equiv (x.insertRight x') x
  -/
  rw [← neg_equiv_neg_iff, ← neg_insertLeft_neg]
  /-
    x x' : SetTheory.PGame
    h : x.LF x'
    ⊢ HasEquiv.Equiv ((Neg.neg x).insertLeft (Neg.neg x')) (Neg.neg x)
  -/
  exact insertLeft_equiv_of_lf (neg_lf_neg_iff.mpr h)
  /-
    🎉 no goals
  -/


/-- Inserting on the left and right commutes. -/
theorem insertRight_insertLeft {x x' x'' : PGame} :
    insertRight (insertLeft x x') x'' = insertLeft (insertRight x x'') x' := by
  /-
    x x' x'' : SetTheory.PGame
    ⊢ Eq ((x.insertLeft x').insertRight x'') ((x.insertRight x'').insertLeft x')
  -/
  cases x; cases x'; cases x''
  /-
    case mk.mk.mk
    α✝² β✝² : Type u_1
    a✝⁵ : α✝² → SetTheory.PGame
    a✝⁴ : β✝² → SetTheory.PGame
    α✝¹ β✝¹ : Type u_1
    a✝³ : α✝¹ → SetTheory.PGame
    a✝² : β✝¹ → SetTheory.PGame
    α✝ β✝ : Type u_1
    a✝¹ : α✝ → SetTheory.PGame
    a✝ : β✝ → SetTheory.PGame
    ⊢ Eq (((SetTheory.PGame.mk α✝² β✝² a✝⁵ a✝⁴).insertLeft (SetTheory.PGame.mk α✝¹ …
  -/
  dsimp [insertLeft, insertRight]
  /-
    🎉 no goals
  -/


/-- The pre-game `star`, which is fuzzy with zero. -/
def star : PGame.{u} :=
  ⟨PUnit, PUnit, fun _ => 0, fun _ => 0⟩


@[simp]
theorem star_leftMoves : star.LeftMoves = PUnit :=
  rfl


@[simp]
theorem star_rightMoves : star.RightMoves = PUnit :=
  rfl


@[simp]
theorem star_moveLeft (x) : star.moveLeft x = 0 :=
  rfl


@[simp]
theorem star_moveRight (x) : star.moveRight x = 0 :=
  rfl


instance uniqueStarLeftMoves : Unique star.LeftMoves :=
  PUnit.instUnique


instance uniqueStarRightMoves : Unique star.RightMoves :=
  PUnit.instUnique


theorem zero_lf_star : 0 ⧏ star := by
  /-
    ⊢ SetTheory.PGame.LF 0 SetTheory.PGame.star
  -/
  rw [zero_lf]
  /-
    ⊢ Exists fun i => ∀ (j : (SetTheory.PGame.star.moveLeft i).RightMoves), SetThe …
  -/
  use default
  /-
    case h
    ⊢ ∀ (j : (SetTheory.PGame.star.moveLeft Inhabited.default).RightMoves), SetThe …
  -/
  rintro ⟨⟩
  /-
    🎉 no goals
  -/


theorem star_lf_zero : star ⧏ 0 := by
  /-
    ⊢ SetTheory.PGame.star.LF 0
  -/
  rw [lf_zero]
  /-
    ⊢ Exists fun j => ∀ (i : (SetTheory.PGame.star.moveRight j).LeftMoves), ((SetT …
  -/
  use default
  /-
    case h
    ⊢ ∀ (i : (SetTheory.PGame.star.moveRight Inhabited.default).LeftMoves), ((SetT …
  -/
  rintro ⟨⟩
  /-
    🎉 no goals
  -/


theorem star_fuzzy_zero : star ‖ 0 :=
  ⟨star_lf_zero, zero_lf_star⟩


@[simp]
                                      /-
                                        ⊢ Eq (Neg.neg SetTheory.PGame.star) SetTheory.PGame.star
                                      -/
theorem neg_star : -star = star := by simp [star]
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
protected theorem zero_lt_one : (0 : PGame) < 1 :=
  lt_of_le_of_lf (zero_le_of_isEmpty_rightMoves 1) (zero_lf_le.2 ⟨default, le_rfl⟩)


/-- The pre-game `up` -/
def up : PGame.{u} :=
  ⟨PUnit, PUnit, fun _ => 0, fun _ => star⟩


@[simp]
theorem up_leftMoves : up.LeftMoves = PUnit :=
  rfl


@[simp]
theorem up_rightMoves : up.RightMoves = PUnit :=
  rfl


@[simp]
theorem up_moveLeft (x) : up.moveLeft x = 0 :=
  rfl


@[simp]
theorem up_moveRight (x) : up.moveRight x = star :=
  rfl


@[simp]
theorem up_neg : 0 < up := by
  /-
    ⊢ LT.lt 0 SetTheory.PGame.up
  -/
  rw [lt_iff_le_and_lf, zero_lf]
  /-
    ⊢ And (LE.le 0 SetTheory.PGame.up) (Exists fun i => ∀ (j : (SetTheory.PGame.up …
  -/
  simp [zero_le_lf, zero_lf_star]
  /-
    🎉 no goals
  -/


theorem star_fuzzy_up : star ‖ up := by
  /-
    ⊢ SetTheory.PGame.star.Fuzzy SetTheory.PGame.up
  -/
  unfold Fuzzy
  /-
    ⊢ And (SetTheory.PGame.star.LF SetTheory.PGame.up) (SetTheory.PGame.up.LF SetT …
  -/
  simp only [← PGame.not_le]
  /-
    ⊢ And (Not (LE.le SetTheory.PGame.up SetTheory.PGame.star)) (Not (LE.le SetThe …
  -/
  simp [le_iff_forall_lf]
  /-
    🎉 no goals
  -/


/-- The pre-game `down` -/
def down : PGame.{u} :=
  ⟨PUnit, PUnit, fun _ => star, fun _ => 0⟩


@[simp]
theorem down_leftMoves : down.LeftMoves = PUnit :=
  rfl


@[simp]
theorem down_rightMoves : down.RightMoves = PUnit :=
  rfl


@[simp]
theorem down_moveLeft (x) : down.moveLeft x = star :=
  rfl


@[simp]
theorem down_moveRight (x) : down.moveRight x = 0 :=
  rfl


@[simp]
theorem down_neg : down < 0 := by
  /-
    ⊢ LT.lt SetTheory.PGame.down 0
  -/
  rw [lt_iff_le_and_lf, lf_zero]
  /-
    ⊢ And (LE.le SetTheory.PGame.down 0) (Exists fun j => ∀ (i : (SetTheory.PGame. …
  -/
  simp [le_zero_lf, star_lf_zero]
  /-
    🎉 no goals
  -/


@[simp]
                                    /-
                                      ⊢ Eq (Neg.neg SetTheory.PGame.down) SetTheory.PGame.up
                                    -/
theorem neg_down : -down = up := by simp [up, down]
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
                                  /-
                                    ⊢ Eq (Neg.neg SetTheory.PGame.up) SetTheory.PGame.down
                                  -/
theorem neg_up : -up = down := by simp [up, down]
                                  /-
                                    🎉 no goals
                                  -/


theorem star_fuzzy_down : star ‖ down := by
  /-
    ⊢ SetTheory.PGame.star.Fuzzy SetTheory.PGame.down
  -/
  rw [← neg_fuzzy_neg_iff, neg_down, neg_star]
  /-
    ⊢ SetTheory.PGame.star.Fuzzy SetTheory.PGame.up
  -/
  exact star_fuzzy_up
  /-
    🎉 no goals
  -/


instance : ZeroLEOneClass PGame :=
  ⟨PGame.zero_lt_one.le⟩


@[simp]
theorem zero_lf_one : (0 : PGame) ⧏ 1 :=
  PGame.zero_lt_one.lf



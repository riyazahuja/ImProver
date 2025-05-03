/-- The natural map between `Π₀ (i : Σ i, α i), δ i.1 i.2` and `Π₀ i (j : α i), δ i j`. -/
def sigmaCurry [∀ i j, Zero (δ i j)] (f : Π₀ (i : Σ _, _), δ i.1 i.2) :
    Π₀ (i) (j), δ i j where
  toFun := fun i ↦
  { toFun := fun j ↦ f ⟨i, j⟩,
    support' := f.support'.map (fun ⟨m, hm⟩ ↦
      ⟨m.filterMap (fun ⟨i', j'⟩ ↦ if h : i' = i then some <| h.rec j' else none),
        fun j ↦ (hm ⟨i, j⟩).imp_left (fun h ↦ (m.mem_filterMap _).mpr ⟨⟨i, j⟩, h, dif_pos rfl⟩)⟩) }
  support' := f.support'.map (fun ⟨m, hm⟩ ↦
    ⟨m.map Sigma.fst, fun i ↦ Decidable.or_iff_not_imp_left.mpr (fun h ↦ DFinsupp.ext
      (fun j ↦ (hm ⟨i, j⟩).resolve_left (fun H ↦ (Multiset.mem_map.not.mp h) ⟨⟨i, j⟩, H, rfl⟩)))⟩)


@[simp]
theorem sigmaCurry_apply [∀ i j, Zero (δ i j)] (f : Π₀ (i : Σ _, _), δ i.1 i.2) (i : ι) (j : α i) :
    sigmaCurry f i j = f ⟨i, j⟩ :=
  rfl


@[simp]
theorem sigmaCurry_zero [∀ i j, Zero (δ i j)] :
    sigmaCurry (0 : Π₀ (i : Σ _, _), δ i.1 i.2) = 0 :=
  rfl


@[simp]
theorem sigmaCurry_add [∀ i j, AddZeroClass (δ i j)] (f g : Π₀ (i : Σ _, _), δ i.1 i.2) :
    #adaptation_note
    /-- After https://github.com/leanprover/lean4/pull/6024
    we needed to add the `(_ : Π₀ (i) (j), δ i j)` type annotation. -/
    sigmaCurry (f + g) = (sigmaCurry f + sigmaCurry g : Π₀ (i) (j), δ i j) := by
  /-
    ι : Type u
    α : ι → Type u_2
    δ : (i : ι) → α i → Type v
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → (j : α i) → AddZeroClass (δ i j)
    f g : DFinsupp fun i => δ i.fst i.snd
    ⊢ Eq (HAdd.hAdd f g).sigmaCurry (HAdd.hAdd f.sigmaCurry g.sigmaCurry)
  -/
  ext (i j)
  /-
    case h.h
    ι : Type u
    α : ι → Type u_2
    δ : (i : ι) → α i → Type v
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → (j : α i) → AddZeroClass (δ i j)
    f g : DFinsupp fun i => δ i.fst i.snd
    i : ι
    j : α i
    ⊢ Eq (((HAdd.hAdd f g).sigmaCurry i) j) (((HAdd.hAdd f.sigmaCurry g.sigmaCurry …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem sigmaCurry_smul [Monoid γ] [∀ i j, AddMonoid (δ i j)] [∀ i j, DistribMulAction γ (δ i j)]
    (r : γ) (f : Π₀ (i : Σ _, _), δ i.1 i.2) :
    #adaptation_note
    /-- After https://github.com/leanprover/lean4/pull/6024
    we needed to add the `(_ : Π₀ (i) (j), δ i j)` type annotation. -/
    sigmaCurry (r • f) = (r • sigmaCurry f : Π₀ (i) (j), δ i j) := by
  /-
    ι : Type u
    γ : Type w
    α : ι → Type u_2
    δ : (i : ι) → α i → Type v
    inst✝³ : DecidableEq ι
    inst✝² : Monoid γ
    inst✝¹ : (i : ι) → (j : α i) → AddMonoid (δ i j)
    inst✝ : (i : ι) → (j : α i) → DistribMulAction γ (δ i j)
    r : γ
    f : DFinsupp fun i => δ i.fst i.snd
    ⊢ Eq (HSMul.hSMul r f).sigmaCurry (HSMul.hSMul r f.sigmaCurry)
  -/
  ext (i j)
  /-
    case h.h
    ι : Type u
    γ : Type w
    α : ι → Type u_2
    δ : (i : ι) → α i → Type v
    inst✝³ : DecidableEq ι
    inst✝² : Monoid γ
    inst✝¹ : (i : ι) → (j : α i) → AddMonoid (δ i j)
    inst✝ : (i : ι) → (j : α i) → DistribMulAction γ (δ i j)
    r : γ
    f : DFinsupp fun i => δ i.fst i.snd
    i : ι
    j : α i
    ⊢ Eq (((HSMul.hSMul r f).sigmaCurry i) j) (((HSMul.hSMul r f.sigmaCurry) i) j)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem sigmaCurry_single [∀ i, DecidableEq (α i)] [∀ i j, Zero (δ i j)]
    (ij : Σ i, α i) (x : δ ij.1 ij.2) :
    sigmaCurry (single ij x) = single ij.1 (single ij.2 x : Π₀ j, δ ij.1 j) := by
  /-
    ι : Type u
    α : ι → Type u_2
    δ : (i : ι) → α i → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → DecidableEq (α i)
    inst✝ : (i : ι) → (j : α i) → Zero (δ i j)
    ij : Sigma fun i => α i
    x : δ ij.fst ij.snd
    ⊢ Eq (DFinsupp.single ij x).sigmaCurry (DFinsupp.single ij.fst (DFinsupp.singl …
  -/
  obtain ⟨i, j⟩ := ij
  /-
    case mk
    ι : Type u
    α : ι → Type u_2
    δ : (i : ι) → α i → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → DecidableEq (α i)
    inst✝ : (i : ι) → (j : α i) → Zero (δ i j)
    i : ι
    j : α i
    x : δ ⟨i, j⟩.fst ⟨i, j⟩.snd
    ⊢ Eq (DFinsupp.single ⟨i, j⟩ x).sigmaCurry (DFinsupp.single ⟨i, j⟩.fst (DFinsu …
  -/
  ext i' j'
  /-
    case mk.h.h
    ι : Type u
    α : ι → Type u_2
    δ : (i : ι) → α i → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → DecidableEq (α i)
    inst✝ : (i : ι) → (j : α i) → Zero (δ i j)
    i : ι
    j : α i
    x : δ ⟨i, j⟩.fst ⟨i, j⟩.snd
    i' : ι
    j' : α i'
    ⊢ Eq (((DFinsupp.single ⟨i, j⟩ x).sigmaCurry i') j') (((DFinsupp.single ⟨i, j⟩ …
  -/
  dsimp only
  /-
    case mk.h.h
    ι : Type u
    α : ι → Type u_2
    δ : (i : ι) → α i → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → DecidableEq (α i)
    inst✝ : (i : ι) → (j : α i) → Zero (δ i j)
    i : ι
    j : α i
    x : δ ⟨i, j⟩.fst ⟨i, j⟩.snd
    i' : ι
    j' : α i'
    ⊢ Eq (((DFinsupp.single ⟨i, j⟩ x).sigmaCurry i') j') (((DFinsupp.single i (DFi …
  -/
  rw [sigmaCurry_apply]
  /-
    case mk.h.h
    ι : Type u
    α : ι → Type u_2
    δ : (i : ι) → α i → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → DecidableEq (α i)
    inst✝ : (i : ι) → (j : α i) → Zero (δ i j)
    i : ι
    j : α i
    x : δ ⟨i, j⟩.fst ⟨i, j⟩.snd
    i' : ι
    j' : α i'
    ⊢ Eq ((DFinsupp.single ⟨i, j⟩ x) ⟨i', j'⟩) (((DFinsupp.single i (DFinsupp.sing …
  -/
  obtain rfl | hi := eq_or_ne i i'
    /-
      case mk.h.h.inl
      ι : Type u
      α : ι → Type u_2
      δ : (i : ι) → α i → Type v
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → DecidableEq (α i)
      inst✝ : (i : ι) → (j : α i) → Zero (δ i j)
      i : ι
      j : α i
      x : δ ⟨i, j⟩.fst ⟨i, j⟩.snd
      j' : α i
      ⊢ Eq ((DFinsupp.single ⟨i, j⟩ x) ⟨i, j'⟩) (((DFinsupp.single i (DFinsupp.singl …
    -/
  · rw [single_eq_same]
    /-
      case mk.h.h.inl
      ι : Type u
      α : ι → Type u_2
      δ : (i : ι) → α i → Type v
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → DecidableEq (α i)
      inst✝ : (i : ι) → (j : α i) → Zero (δ i j)
      i : ι
      j : α i
      x : δ ⟨i, j⟩.fst ⟨i, j⟩.snd
      j' : α i
      ⊢ Eq ((DFinsupp.single ⟨i, j⟩ x) ⟨i, j'⟩) ((DFinsupp.single j x) j')
    -/
    obtain rfl | hj := eq_or_ne j j'
      /-
        case mk.h.h.inl.inl
        ι : Type u
        α : ι → Type u_2
        δ : (i : ι) → α i → Type v
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → DecidableEq (α i)
        inst✝ : (i : ι) → (j : α i) → Zero (δ i j)
        i : ι
        j : α i
        x : δ ⟨i, j⟩.fst ⟨i, j⟩.snd
        ⊢ Eq ((DFinsupp.single ⟨i, j⟩ x) ⟨i, j⟩) ((DFinsupp.single j x) j)
      -/
    · rw [single_eq_same, single_eq_same]
      /-
        🎉 no goals
      -/
      /-
        case mk.h.h.inl.inr
        ι : Type u
        α : ι → Type u_2
        δ : (i : ι) → α i → Type v
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → DecidableEq (α i)
        inst✝ : (i : ι) → (j : α i) → Zero (δ i j)
        i : ι
        j : α i
        x : δ ⟨i, j⟩.fst ⟨i, j⟩.snd
        j' : α i
        hj : Ne j j'
        ⊢ Eq ((DFinsupp.single ⟨i, j⟩ x) ⟨i, j'⟩) ((DFinsupp.single j x) j')
      -/
    · rw [single_eq_of_ne, single_eq_of_ne hj]
      /-
        case mk.h.h.inl.inr
        ι : Type u
        α : ι → Type u_2
        δ : (i : ι) → α i → Type v
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → DecidableEq (α i)
        inst✝ : (i : ι) → (j : α i) → Zero (δ i j)
        i : ι
        j : α i
        x : δ ⟨i, j⟩.fst ⟨i, j⟩.snd
        j' : α i
        hj : Ne j j'
        ⊢ Ne ⟨i, j⟩ ⟨i, j'⟩
      -/
      simpa using hj
      /-
        🎉 no goals
      -/
    /-
      case mk.h.h.inr
      ι : Type u
      α : ι → Type u_2
      δ : (i : ι) → α i → Type v
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → DecidableEq (α i)
      inst✝ : (i : ι) → (j : α i) → Zero (δ i j)
      i : ι
      j : α i
      x : δ ⟨i, j⟩.fst ⟨i, j⟩.snd
      i' : ι
      j' : α i'
      hi : Ne i i'
      ⊢ Eq ((DFinsupp.single ⟨i, j⟩ x) ⟨i', j'⟩) (((DFinsupp.single i (DFinsupp.sing …
    -/
  · rw [single_eq_of_ne, single_eq_of_ne hi, zero_apply]
    /-
      case mk.h.h.inr
      ι : Type u
      α : ι → Type u_2
      δ : (i : ι) → α i → Type v
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → DecidableEq (α i)
      inst✝ : (i : ι) → (j : α i) → Zero (δ i j)
      i : ι
      j : α i
      x : δ ⟨i, j⟩.fst ⟨i, j⟩.snd
      i' : ι
      j' : α i'
      hi : Ne i i'
      ⊢ Ne ⟨i, j⟩ ⟨i', j'⟩
    -/
    simp [hi]
    /-
      🎉 no goals
    -/

/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/

/-- The natural map between `Π₀ i (j : α i), δ i j` and `Π₀ (i : Σ i, α i), δ i.1 i.2`, inverse of
`curry`. -/
def sigmaUncurry [∀ i j, Zero (δ i j)] [DecidableEq ι] (f : Π₀ (i) (j), δ i j) :
    Π₀ i : Σ_, _, δ i.1 i.2 where
  toFun i := f i.1 i.2
  support' :=
    f.support'.bind fun s =>
      (Trunc.finChoice (fun i : ↥s.val.toFinset => (f i).support')).map fun fs =>
        ⟨s.val.toFinset.attach.val.bind fun i => (fs i).val.map (Sigma.mk i.val), by
          /-
            ι : Type u
            γ : Type w
            β : ι → Type v
            β₁ : ι → Type v₁
            β₂ : ι → Type v₂
            κ : Type u_1
            α : ι → Type u_2
            δ : (i : ι) → α i → Type v
            inst✝² : DecidableEq ι
            inst✝¹ : (i : ι) → (j : α i) → Zero (δ i j)
            inst✝ : DecidableEq ι
            f : DFinsupp fun i => DFinsupp fun j => δ i j
            s : Subtype fun s => ∀ (i : ι), Or (Membership.mem s i) (Eq (f.toFun i) 0)
            fs : (i : Subtype fun x => Membership.mem (↑s).toFinset x) → Subtype fun s_1 = …
            ⊢ ∀ (i : Sigma fun x => α x), Or (Membership.mem ((↑s).toFinset.attach.val.bin …
          -/
          rintro ⟨i, a⟩
          cases s.prop i with
          | inl hi =>
            cases (fs ⟨i, Multiset.mem_toFinset.mpr hi⟩).prop a with
            | inl ha =>
              left; rw [Multiset.mem_bind]
              use ⟨i, Multiset.mem_toFinset.mpr hi⟩
              constructor
              case right => simp [ha]
              case left => apply Multiset.mem_attach
            | inr ha => right; simp [toFun_eq_coe (f i) ▸ ha]
          | inr hi => right; simp [toFun_eq_coe f ▸ hi]⟩

/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/

@[simp]
theorem sigmaUncurry_apply [∀ i j, Zero (δ i j)]
    (f : Π₀ (i) (j), δ i j) (i : ι) (j : α i) :
    sigmaUncurry f ⟨i, j⟩ = f i j :=
  rfl

/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/

@[simp]
theorem sigmaUncurry_zero [∀ i j, Zero (δ i j)] :
    sigmaUncurry (0 : Π₀ (i) (j), δ i j) = 0 :=
  rfl

/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/

@[simp]
theorem sigmaUncurry_add [∀ i j, AddZeroClass (δ i j)] (f g : Π₀ (i) (j), δ i j) :
    sigmaUncurry (f + g) = sigmaUncurry f + sigmaUncurry g :=
  DFunLike.coe_injective rfl

/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/

@[simp]
theorem sigmaUncurry_smul [Monoid γ] [∀ i j, AddMonoid (δ i j)]
    [∀ i j, DistribMulAction γ (δ i j)]
    (r : γ) (f : Π₀ (i) (j), δ i j) : sigmaUncurry (r • f) = r • sigmaUncurry f :=
  DFunLike.coe_injective rfl


@[simp]
theorem sigmaUncurry_single [∀ i j, Zero (δ i j)] [∀ i, DecidableEq (α i)]
    (i) (j : α i) (x : δ i j) :
                                                                                 /-
                                                                                   ι : Type u
                                                                                   γ : Type w
                                                                                   β : ι → Type v
                                                                                   β₁ : ι → Type v₁
                                                                                   β₂ : ι → Type v₂
                                                                                   κ : Type u_1
                                                                                   α : ι → Type u_2
                                                                                   δ : (i : ι) → α i → Type v
                                                                                   inst✝² : DecidableEq ι
                                                                                   inst✝¹ : (i : ι) → (j : α i) → Zero (δ i j)
                                                                                   inst✝ : (i : ι) → DecidableEq (α i)
                                                                                   i : ι
                                                                                   j : α i
                                                                                   x : δ i j
                                                                                   ⊢ δ ⟨i, j⟩.fst ⟨i, j⟩.snd
                                                                                 -/
    sigmaUncurry (single i (single j x : Π₀ j : α i, δ i j)) = single ⟨i, j⟩ (by exact x) := by
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
  /-
    ι : Type u
    α : ι → Type u_2
    δ : (i : ι) → α i → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → (j : α i) → Zero (δ i j)
    inst✝ : (i : ι) → DecidableEq (α i)
    i : ι
    j : α i
    x : δ i j
    ⊢ Eq (DFinsupp.single i (DFinsupp.single j x)).sigmaUncurry (DFinsupp.single ⟨ …
  -/
  ext ⟨i', j'⟩
  /-
    case h.mk
    ι : Type u
    α : ι → Type u_2
    δ : (i : ι) → α i → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → (j : α i) → Zero (δ i j)
    inst✝ : (i : ι) → DecidableEq (α i)
    i : ι
    j : α i
    x : δ i j
    i' : ι
    j' : α i'
    ⊢ Eq ((DFinsupp.single i (DFinsupp.single j x)).sigmaUncurry ⟨i', j'⟩) ((DFins …
  -/
  dsimp only
  /-
    case h.mk
    ι : Type u
    α : ι → Type u_2
    δ : (i : ι) → α i → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → (j : α i) → Zero (δ i j)
    inst✝ : (i : ι) → DecidableEq (α i)
    i : ι
    j : α i
    x : δ i j
    i' : ι
    j' : α i'
    ⊢ Eq ((DFinsupp.single i (DFinsupp.single j x)).sigmaUncurry ⟨i', j'⟩) ((DFins …
  -/
  rw [sigmaUncurry_apply]
  /-
    case h.mk
    ι : Type u
    α : ι → Type u_2
    δ : (i : ι) → α i → Type v
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → (j : α i) → Zero (δ i j)
    inst✝ : (i : ι) → DecidableEq (α i)
    i : ι
    j : α i
    x : δ i j
    i' : ι
    j' : α i'
    ⊢ Eq (((DFinsupp.single i (DFinsupp.single j x)) i') j') ((DFinsupp.single ⟨i, …
  -/
  obtain rfl | hi := eq_or_ne i i'
    /-
      case h.mk.inl
      ι : Type u
      α : ι → Type u_2
      δ : (i : ι) → α i → Type v
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → (j : α i) → Zero (δ i j)
      inst✝ : (i : ι) → DecidableEq (α i)
      i : ι
      j : α i
      x : δ i j
      j' : α i
      ⊢ Eq (((DFinsupp.single i (DFinsupp.single j x)) i) j') ((DFinsupp.single ⟨i,  …
    -/
  · rw [single_eq_same]
    /-
      case h.mk.inl
      ι : Type u
      α : ι → Type u_2
      δ : (i : ι) → α i → Type v
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → (j : α i) → Zero (δ i j)
      inst✝ : (i : ι) → DecidableEq (α i)
      i : ι
      j : α i
      x : δ i j
      j' : α i
      ⊢ Eq ((DFinsupp.single j x) j') ((DFinsupp.single ⟨i, j⟩ x) ⟨i, j'⟩)
    -/
    obtain rfl | hj := eq_or_ne j j'
      /-
        case h.mk.inl.inl
        ι : Type u
        α : ι → Type u_2
        δ : (i : ι) → α i → Type v
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → (j : α i) → Zero (δ i j)
        inst✝ : (i : ι) → DecidableEq (α i)
        i : ι
        j : α i
        x : δ i j
        ⊢ Eq ((DFinsupp.single j x) j) ((DFinsupp.single ⟨i, j⟩ x) ⟨i, j⟩)
      -/
    · rw [single_eq_same, single_eq_same]
      /-
        🎉 no goals
      -/
      /-
        case h.mk.inl.inr
        ι : Type u
        α : ι → Type u_2
        δ : (i : ι) → α i → Type v
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → (j : α i) → Zero (δ i j)
        inst✝ : (i : ι) → DecidableEq (α i)
        i : ι
        j : α i
        x : δ i j
        j' : α i
        hj : Ne j j'
        ⊢ Eq ((DFinsupp.single j x) j') ((DFinsupp.single ⟨i, j⟩ x) ⟨i, j'⟩)
      -/
    · rw [single_eq_of_ne hj, single_eq_of_ne]
      /-
        case h.mk.inl.inr
        ι : Type u
        α : ι → Type u_2
        δ : (i : ι) → α i → Type v
        inst✝² : DecidableEq ι
        inst✝¹ : (i : ι) → (j : α i) → Zero (δ i j)
        inst✝ : (i : ι) → DecidableEq (α i)
        i : ι
        j : α i
        x : δ i j
        j' : α i
        hj : Ne j j'
        ⊢ Ne ⟨i, j⟩ ⟨i, j'⟩
      -/
      simpa using hj
      /-
        🎉 no goals
      -/
    /-
      case h.mk.inr
      ι : Type u
      α : ι → Type u_2
      δ : (i : ι) → α i → Type v
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → (j : α i) → Zero (δ i j)
      inst✝ : (i : ι) → DecidableEq (α i)
      i : ι
      j : α i
      x : δ i j
      i' : ι
      j' : α i'
      hi : Ne i i'
      ⊢ Eq (((DFinsupp.single i (DFinsupp.single j x)) i') j') ((DFinsupp.single ⟨i, …
    -/
  · rw [single_eq_of_ne hi, single_eq_of_ne, zero_apply]
    /-
      case h.mk.inr
      ι : Type u
      α : ι → Type u_2
      δ : (i : ι) → α i → Type v
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → (j : α i) → Zero (δ i j)
      inst✝ : (i : ι) → DecidableEq (α i)
      i : ι
      j : α i
      x : δ i j
      i' : ι
      j' : α i'
      hi : Ne i i'
      ⊢ Ne ⟨i, j⟩ ⟨i', j'⟩
    -/
    simp [hi]
    /-
      🎉 no goals
    -/

/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i j) -/

/-- The natural bijection between `Π₀ (i : Σ i, α i), δ i.1 i.2` and `Π₀ i (j : α i), δ i j`.

This is the dfinsupp version of `Equiv.piCurry`. -/
def sigmaCurryEquiv [∀ i j, Zero (δ i j)] [DecidableEq ι] :
    (Π₀ i : Σ_, _, δ i.1 i.2) ≃ Π₀ (i) (j), δ i j where
  toFun := sigmaCurry
  invFun := sigmaUncurry
  left_inv f := by
    /-
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      κ : Type u_1
      α : ι → Type u_2
      δ : (i : ι) → α i → Type v
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → (j : α i) → Zero (δ i j)
      inst✝ : DecidableEq ι
      f : DFinsupp fun i => δ i.fst i.snd
      ⊢ Eq f.sigmaCurry.sigmaUncurry f
    -/
    ext ⟨i, j⟩
    /-
      case h.mk
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      κ : Type u_1
      α : ι → Type u_2
      δ : (i : ι) → α i → Type v
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → (j : α i) → Zero (δ i j)
      inst✝ : DecidableEq ι
      f : DFinsupp fun i => δ i.fst i.snd
      i : ι
      j : α i
      ⊢ Eq (f.sigmaCurry.sigmaUncurry ⟨i, j⟩) (f ⟨i, j⟩)
    -/
    rw [sigmaUncurry_apply, sigmaCurry_apply]
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      κ : Type u_1
      α : ι → Type u_2
      δ : (i : ι) → α i → Type v
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → (j : α i) → Zero (δ i j)
      inst✝ : DecidableEq ι
      f : DFinsupp fun i => DFinsupp fun j => δ i j
      ⊢ Eq f.sigmaUncurry.sigmaCurry f
    -/
    ext i j
    /-
      case h.h
      ι : Type u
      γ : Type w
      β : ι → Type v
      β₁ : ι → Type v₁
      β₂ : ι → Type v₂
      κ : Type u_1
      α : ι → Type u_2
      δ : (i : ι) → α i → Type v
      inst✝² : DecidableEq ι
      inst✝¹ : (i : ι) → (j : α i) → Zero (δ i j)
      inst✝ : DecidableEq ι
      f : DFinsupp fun i => DFinsupp fun j => δ i j
      i : ι
      j : α i
      ⊢ Eq ((f.sigmaUncurry.sigmaCurry i) j) ((f i) j)
    -/
    rw [sigmaCurry_apply, sigmaUncurry_apply]
    /-
      🎉 no goals
    -/



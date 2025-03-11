/-- This structure contains the information left for a local trivialization (which is implemented
below as `Trivialization F proj`) if the total space has not been given a topology, but we
have a topology on both the fiber and the base space. Through the construction
`topological_fiber_prebundle F proj` it will be possible to promote a
`Pretrivialization F proj` to a `Trivialization F proj`. -/
structure Pretrivialization (proj : Z → B) extends PartialEquiv Z (B × F) where
  open_target : IsOpen target
  baseSet : Set B
  open_baseSet : IsOpen baseSet
  source_eq : source = proj ⁻¹' baseSet
  target_eq : target = baseSet ×ˢ univ
  proj_toFun : ∀ p ∈ source, (toFun p).1 = proj p


/-- Coercion of a pretrivialization to a function. We don't use `e.toFun` in the `CoeFun` instance
because it is actually `e.toPartialEquiv.toFun`, so `simp` will apply lemmas about
`toPartialEquiv`. While we may want to switch to this behavior later, doing it mid-port will break a
lot of proofs. -/
@[coe] def toFun' : Z → (B × F) := e.toFun


instance : CoeFun (Pretrivialization F proj) fun _ => Z → B × F := ⟨toFun'⟩


@[ext]
lemma ext' (e e' : Pretrivialization F proj) (h₁ : e.toPartialEquiv = e'.toPartialEquiv)
    (h₂ : e.baseSet = e'.baseSet) : e = e' := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    proj : Z → B
    e e' : Pretrivialization F proj
    h₁ : Eq e.toPartialEquiv e'.toPartialEquiv
    h₂ : Eq e.baseSet e'.baseSet
    ⊢ Eq e e'
  -/
  cases e; cases e'; congr
                     /-
                       🎉 no goals
                     -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: move `ext` here?

lemma ext {e e' : Pretrivialization F proj} (h₁ : ∀ x, e x = e' x)
    (h₂ : ∀ x, e.toPartialEquiv.symm x = e'.toPartialEquiv.symm x) (h₃ : e.baseSet = e'.baseSet) :
    e = e' := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    proj : Z → B
    e e' : Pretrivialization F proj
    h₁ : ∀ (x : Z), Eq (↑e x) (↑e' x)
    h₂ : ∀ (x : Prod B F), Eq (↑e.symm x) (↑e'.symm x)
    h₃ : Eq e.baseSet e'.baseSet
    ⊢ Eq e e'
  -/
  ext1 <;> [ext1; exact h₃]
    /-
      case h₁.h
      B : Type u_1
      F : Type u_2
      Z : Type u_4
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      proj : Z → B
      e e' : Pretrivialization F proj
      h₁ : ∀ (x : Z), Eq (↑e x) (↑e' x)
      h₂ : ∀ (x : Prod B F), Eq (↑e.symm x) (↑e'.symm x)
      h₃ : Eq e.baseSet e'.baseSet
      x✝ : Z
      ⊢ Eq (↑e.toPartialEquiv x✝) (↑e'.toPartialEquiv x✝)
    -/
  · apply h₁
    /-
      🎉 no goals
    -/
    /-
      case h₁.hsymm
      B : Type u_1
      F : Type u_2
      Z : Type u_4
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      proj : Z → B
      e e' : Pretrivialization F proj
      h₁ : ∀ (x : Z), Eq (↑e x) (↑e' x)
      h₂ : ∀ (x : Prod B F), Eq (↑e.symm x) (↑e'.symm x)
      h₃ : Eq e.baseSet e'.baseSet
      x✝ : Prod B F
      ⊢ Eq (↑e.symm x✝) (↑e'.symm x✝)
    -/
  · apply h₂
    /-
      🎉 no goals
    -/
    /-
      case h₁.hs
      B : Type u_1
      F : Type u_2
      Z : Type u_4
      inst✝¹ : TopologicalSpace B
      inst✝ : TopologicalSpace F
      proj : Z → B
      e e' : Pretrivialization F proj
      h₁ : ∀ (x : Z), Eq (↑e x) (↑e' x)
      h₂ : ∀ (x : Prod B F), Eq (↑e.symm x) (↑e'.symm x)
      h₃ : Eq e.baseSet e'.baseSet
      ⊢ Eq e.source e'.source
    -/
  · rw [e.source_eq, e'.source_eq, h₃]
    /-
      🎉 no goals
    -/


/-- If the fiber is nonempty, then the projection also is. -/
lemma toPartialEquiv_injective [Nonempty F] :
    Injective (toPartialEquiv : Pretrivialization F proj → PartialEquiv Z (B × F)) := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    proj : Z → B
    inst✝ : Nonempty F
    ⊢ Function.Injective Pretrivialization.toPartialEquiv
  -/
  refine fun e e' h ↦ ext' _ _ h ?_
  simpa only [fst_image_prod, univ_nonempty, target_eq]
    using congr_arg (Prod.fst '' PartialEquiv.target ·) h


@[simp, mfld_simps]
theorem coe_coe : ⇑e.toPartialEquiv = e :=
  rfl


@[simp, mfld_simps]
theorem coe_fst (ex : x ∈ e.source) : (e x).1 = proj x :=
  e.proj_toFun x ex


                                                             /-
                                                               B : Type u_1
                                                               F : Type u_2
                                                               Z : Type u_4
                                                               inst✝¹ : TopologicalSpace B
                                                               inst✝ : TopologicalSpace F
                                                               proj : Z → B
                                                               e : Pretrivialization F proj
                                                               x : Z
                                                               ⊢ Iff (Membership.mem e.source x) (Membership.mem e.baseSet (proj x))
                                                             -/
theorem mem_source : x ∈ e.source ↔ proj x ∈ e.baseSet := by rw [e.source_eq, mem_preimage]
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem coe_fst' (ex : proj x ∈ e.baseSet) : (e x).1 = proj x :=
  e.coe_fst (e.mem_source.2 ex)


protected theorem eqOn : EqOn (Prod.fst ∘ e) proj e.source := fun _ hx => e.coe_fst hx


theorem mk_proj_snd (ex : x ∈ e.source) : (proj x, (e x).2) = e x :=
  Prod.ext (e.coe_fst ex).symm rfl


theorem mk_proj_snd' (ex : proj x ∈ e.baseSet) : (proj x, (e x).2) = e x :=
  Prod.ext (e.coe_fst' ex).symm rfl


/-- Composition of inverse and coercion from the subtype of the target. -/
def setSymm : e.target → Z :=
  e.target.restrict e.toPartialEquiv.symm


theorem mem_target {x : B × F} : x ∈ e.target ↔ x.1 ∈ e.baseSet := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    proj : Z → B
    e : Pretrivialization F proj
    x : Prod B F
    ⊢ Iff (Membership.mem e.target x) (Membership.mem e.baseSet x.1)
  -/
  rw [e.target_eq, prod_univ, mem_preimage]
  /-
    🎉 no goals
  -/


theorem proj_symm_apply {x : B × F} (hx : x ∈ e.target) : proj (e.toPartialEquiv.symm x) = x.1 := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    proj : Z → B
    e : Pretrivialization F proj
    x : Prod B F
    hx : Membership.mem e.target x
    ⊢ Eq (proj (↑e.symm x)) x.1
  -/
  have := (e.coe_fst (e.map_target hx)).symm
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    proj : Z → B
    e : Pretrivialization F proj
    x : Prod B F
    hx : Membership.mem e.target x
    this : Eq (proj (↑e.symm x)) (↑e (↑e.symm x)).1
    ⊢ Eq (proj (↑e.symm x)) x.1
  -/
  rwa [← e.coe_coe, e.right_inv hx] at this
  /-
    🎉 no goals
  -/


theorem proj_symm_apply' {b : B} {x : F} (hx : b ∈ e.baseSet) :
    proj (e.toPartialEquiv.symm (b, x)) = b :=
  e.proj_symm_apply (e.mem_target.2 hx)


theorem proj_surjOn_baseSet [Nonempty F] : Set.SurjOn proj e.source e.baseSet := fun b hb =>
  let ⟨y⟩ := ‹Nonempty F›
  ⟨e.toPartialEquiv.symm (b, y), e.toPartialEquiv.map_target <| e.mem_target.2 hb,
    e.proj_symm_apply' hb⟩


theorem apply_symm_apply {x : B × F} (hx : x ∈ e.target) : e (e.toPartialEquiv.symm x) = x :=
  e.toPartialEquiv.right_inv hx


theorem apply_symm_apply' {b : B} {x : F} (hx : b ∈ e.baseSet) :
    e (e.toPartialEquiv.symm (b, x)) = (b, x) :=
  e.apply_symm_apply (e.mem_target.2 hx)


theorem symm_apply_apply {x : Z} (hx : x ∈ e.source) : e.toPartialEquiv.symm (e x) = x :=
  e.toPartialEquiv.left_inv hx


@[simp, mfld_simps]
theorem symm_apply_mk_proj {x : Z} (ex : x ∈ e.source) :
    e.toPartialEquiv.symm (proj x, (e x).2) = x := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    proj : Z → B
    e : Pretrivialization F proj
    x : Z
    ex : Membership.mem e.source x
    ⊢ Eq (↑e.symm { fst := proj x, snd := (↑e x).2 }) x
  -/
  rw [← e.coe_fst ex, ← e.coe_coe, e.left_inv ex]
  /-
    🎉 no goals
  -/


@[simp, mfld_simps]
theorem preimage_symm_proj_baseSet :
    e.toPartialEquiv.symm ⁻¹' (proj ⁻¹' e.baseSet) ∩ e.target = e.target := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    proj : Z → B
    e : Pretrivialization F proj
    ⊢ Eq (Inter.inter (Set.preimage (↑e.symm) (Set.preimage proj e.baseSet)) e.tar …
  -/
  refine inter_eq_right.mpr fun x hx => ?_
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    proj : Z → B
    e : Pretrivialization F proj
    x : Prod B F
    hx : Membership.mem e.target x
    ⊢ Membership.mem (Set.preimage (↑e.symm) (Set.preimage proj e.baseSet)) x
  -/
  simp only [mem_preimage, PartialEquiv.invFun_as_coe, e.proj_symm_apply hx]
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    proj : Z → B
    e : Pretrivialization F proj
    x : Prod B F
    hx : Membership.mem e.target x
    ⊢ Membership.mem e.baseSet x.1
  -/
  exact e.mem_target.mp hx
  /-
    🎉 no goals
  -/


@[simp, mfld_simps]
theorem preimage_symm_proj_inter (s : Set B) :
    e.toPartialEquiv.symm ⁻¹' (proj ⁻¹' s) ∩ e.baseSet ×ˢ univ = (s ∩ e.baseSet) ×ˢ univ := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    proj : Z → B
    e : Pretrivialization F proj
    s : Set B
    ⊢ Eq (Inter.inter (Set.preimage (↑e.symm) (Set.preimage proj s)) (SProd.sprod  …
  -/
  ext ⟨x, y⟩
  suffices x ∈ e.baseSet → (proj (e.toPartialEquiv.symm (x, y)) ∈ s ↔ x ∈ s) by
    simpa only [prod_mk_mem_set_prod_eq, mem_inter_iff, and_true, mem_univ, and_congr_left_iff]
  /-
    case h.mk
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    proj : Z → B
    e : Pretrivialization F proj
    s : Set B
    x : B
    y : F
    ⊢ Membership.mem e.baseSet x → Iff (Membership.mem s (proj (↑e.symm { fst := x …
  -/
  intro h
  /-
    case h.mk
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    proj : Z → B
    e : Pretrivialization F proj
    s : Set B
    x : B
    y : F
    h : Membership.mem e.baseSet x
    ⊢ Iff (Membership.mem s (proj (↑e.symm { fst := x, snd := y }))) (Membership.m …
  -/
  rw [e.proj_symm_apply' h]
  /-
    🎉 no goals
  -/


theorem target_inter_preimage_symm_source_eq (e f : Pretrivialization F proj) :
    f.target ∩ f.toPartialEquiv.symm ⁻¹' e.source = (e.baseSet ∩ f.baseSet) ×ˢ univ := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    proj : Z → B
    e f : Pretrivialization F proj
    ⊢ Eq (Inter.inter f.target (Set.preimage (↑f.symm) e.source)) (SProd.sprod (In …
  -/
  rw [inter_comm, f.target_eq, e.source_eq, f.preimage_symm_proj_inter]
  /-
    🎉 no goals
  -/


theorem trans_source (e f : Pretrivialization F proj) :
    (f.toPartialEquiv.symm.trans e.toPartialEquiv).source = (e.baseSet ∩ f.baseSet) ×ˢ univ := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    proj : Z → B
    e f : Pretrivialization F proj
    ⊢ Eq (f.symm.trans e.toPartialEquiv).source (SProd.sprod (Inter.inter e.baseSe …
  -/
  rw [PartialEquiv.trans_source, PartialEquiv.symm_source, e.target_inter_preimage_symm_source_eq]
  /-
    🎉 no goals
  -/


theorem symm_trans_symm (e e' : Pretrivialization F proj) :
    (e.toPartialEquiv.symm.trans e'.toPartialEquiv).symm
      = e'.toPartialEquiv.symm.trans e.toPartialEquiv := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    proj : Z → B
    e e' : Pretrivialization F proj
    ⊢ Eq (e.symm.trans e'.toPartialEquiv).symm (e'.symm.trans e.toPartialEquiv)
  -/
  rw [PartialEquiv.trans_symm_eq_symm_trans_symm, PartialEquiv.symm_symm]
  /-
    🎉 no goals
  -/


theorem symm_trans_source_eq (e e' : Pretrivialization F proj) :
    (e.toPartialEquiv.symm.trans e'.toPartialEquiv).source = (e.baseSet ∩ e'.baseSet) ×ˢ univ := by
  rw [PartialEquiv.trans_source, e'.source_eq, PartialEquiv.symm_source, e.target_eq, inter_comm,
    e.preimage_symm_proj_inter, inter_comm]


theorem symm_trans_target_eq (e e' : Pretrivialization F proj) :
    (e.toPartialEquiv.symm.trans e'.toPartialEquiv).target = (e.baseSet ∩ e'.baseSet) ×ˢ univ := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝¹ : TopologicalSpace B
    inst✝ : TopologicalSpace F
    proj : Z → B
    e e' : Pretrivialization F proj
    ⊢ Eq (e.symm.trans e'.toPartialEquiv).target (SProd.sprod (Inter.inter e.baseS …
  -/
  rw [← PartialEquiv.symm_source, symm_trans_symm, symm_trans_source_eq, inter_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_mem_source : ↑y ∈ e'.source ↔ b ∈ e'.baseSet :=
  e'.mem_source


@[simp, mfld_simps]
theorem coe_coe_fst (hb : b ∈ e'.baseSet) : (e' y).1 = b :=
  e'.coe_fst (e'.mem_source.2 hb)


theorem mk_mem_target {x : B} {y : F} : (x, y) ∈ e'.target ↔ x ∈ e'.baseSet :=
  e'.mem_target


theorem symm_coe_proj {x : B} {y : F} (e' : Pretrivialization F (π F E)) (h : x ∈ e'.baseSet) :
    (e'.toPartialEquiv.symm (x, y)).1 = x :=
  e'.proj_symm_apply' h


open Classical in
/-- A fiberwise inverse to `e`. This is the function `F → E b` that induces a local inverse
`B × F → TotalSpace F E` of `e` on `e.baseSet`. It is defined to be `0` outside `e.baseSet`. -/
protected noncomputable def symm (e : Pretrivialization F (π F E)) (b : B) (y : F) : E b :=
  if hb : b ∈ e.baseSet then
    cast (congr_arg E (e.proj_symm_apply' hb)) (e.toPartialEquiv.symm (b, y)).2
  else 0


theorem symm_apply (e : Pretrivialization F (π F E)) {b : B} (hb : b ∈ e.baseSet) (y : F) :
    e.symm b y = cast (congr_arg E (e.symm_coe_proj hb)) (e.toPartialEquiv.symm (b, y)).2 :=
  dif_pos hb


theorem symm_apply_of_not_mem (e : Pretrivialization F (π F E)) {b : B} (hb : b ∉ e.baseSet)
    (y : F) : e.symm b y = 0 :=
  dif_neg hb


theorem coe_symm_of_not_mem (e : Pretrivialization F (π F E)) {b : B} (hb : b ∉ e.baseSet) :
    (e.symm b : F → E b) = 0 :=
  funext fun _ => dif_neg hb


theorem mk_symm (e : Pretrivialization F (π F E)) {b : B} (hb : b ∈ e.baseSet) (y : F) :
    TotalSpace.mk b (e.symm b y) = e.toPartialEquiv.symm (b, y) := by
  /-
    B : Type u_1
    F : Type u_2
    E : B → Type u_3
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → Zero (E x)
    e : Pretrivialization F Bundle.TotalSpace.proj
    b : B
    hb : Membership.mem e.baseSet b
    y : F
    ⊢ Eq { proj := b, snd := e.symm b y } (↑e.symm { fst := b, snd := y })
  -/
  simp only [e.symm_apply hb, TotalSpace.mk_cast (e.proj_symm_apply' hb), TotalSpace.eta]
  /-
    🎉 no goals
  -/


theorem symm_proj_apply (e : Pretrivialization F (π F E)) (z : TotalSpace F E)
    (hz : z.proj ∈ e.baseSet) : e.symm z.proj (e z).2 = z.2 := by
  /-
    B : Type u_1
    F : Type u_2
    E : B → Type u_3
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → Zero (E x)
    e : Pretrivialization F Bundle.TotalSpace.proj
    z : Bundle.TotalSpace F E
    hz : Membership.mem e.baseSet z.proj
    ⊢ Eq (e.symm z.proj (↑e z).2) z.snd
  -/
  rw [e.symm_apply hz, cast_eq_iff_heq, e.mk_proj_snd' hz, e.symm_apply_apply (e.mem_source.mpr hz)]
  /-
    🎉 no goals
  -/


theorem symm_apply_apply_mk (e : Pretrivialization F (π F E)) {b : B} (hb : b ∈ e.baseSet)
    (y : E b) : e.symm b (e ⟨b, y⟩).2 = y :=
  e.symm_proj_apply ⟨b, y⟩ hb


theorem apply_mk_symm (e : Pretrivialization F (π F E)) {b : B} (hb : b ∈ e.baseSet) (y : F) :
    e ⟨b, e.symm b y⟩ = (b, y) := by
  /-
    B : Type u_1
    F : Type u_2
    E : B → Type u_3
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    inst✝ : (x : B) → Zero (E x)
    e : Pretrivialization F Bundle.TotalSpace.proj
    b : B
    hb : Membership.mem e.baseSet b
    y : F
    ⊢ Eq (↑e { proj := b, snd := e.symm b y }) { fst := b, snd := y }
  -/
  rw [e.mk_symm hb, e.apply_symm_apply (e.mk_mem_target.mpr hb)]
  /-
    🎉 no goals
  -/


/-- A structure extending partial homeomorphisms, defining a local trivialization of a projection
`proj : Z → B` with fiber `F`, as a partial homeomorphism between `Z` and `B × F` defined between
two sets of the form `proj ⁻¹' baseSet` and `baseSet × F`, acting trivially on the first coordinate.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): was @[nolint has_nonempty_instance]
structure Trivialization (proj : Z → B) extends PartialHomeomorph Z (B × F) where
  baseSet : Set B
  open_baseSet : IsOpen baseSet
  source_eq : source = proj ⁻¹' baseSet
  target_eq : target = baseSet ×ˢ univ
  proj_toFun : ∀ p ∈ source, (toPartialHomeomorph p).1 = proj p


@[ext]
lemma ext' (e e' : Trivialization F proj) (h₁ : e.toPartialHomeomorph = e'.toPartialHomeomorph)
    (h₂ : e.baseSet = e'.baseSet) : e = e' := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    proj : Z → B
    inst✝ : TopologicalSpace Z
    e e' : Trivialization F proj
    h₁ : Eq e.toPartialHomeomorph e'.toPartialHomeomorph
    h₂ : Eq e.baseSet e'.baseSet
    ⊢ Eq e e'
  -/
  cases e; cases e'; congr
                     /-
                       🎉 no goals
                     -/


/-- Coercion of a trivialization to a function. We don't use `e.toFun` in the `CoeFun` instance
because it is actually `e.toPartialEquiv.toFun`, so `simp` will apply lemmas about
`toPartialEquiv`. While we may want to switch to this behavior later, doing it mid-port will break a
lot of proofs. -/
@[coe] def toFun' : Z → (B × F) := e.toFun


/-- Natural identification as a `Pretrivialization`. -/
def toPretrivialization : Pretrivialization F proj :=
  { e with }


instance : CoeFun (Trivialization F proj) fun _ => Z → B × F := ⟨toFun'⟩


instance : Coe (Trivialization F proj) (Pretrivialization F proj) :=
  ⟨toPretrivialization⟩


theorem toPretrivialization_injective :
    Function.Injective fun e : Trivialization F proj => e.toPretrivialization := fun e e' h => by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    proj : Z → B
    inst✝ : TopologicalSpace Z
    e e' : Trivialization F proj
    h : Eq ((fun e => e.toPretrivialization) e) ((fun e => e.toPretrivialization)  …
    ⊢ Eq e e'
  -/
  ext1
  exacts [PartialHomeomorph.toPartialEquiv_injective (congr_arg Pretrivialization.toPartialEquiv h),
    congr_arg Pretrivialization.baseSet h]


@[simp, mfld_simps]
theorem coe_coe : ⇑e.toPartialHomeomorph = e :=
  rfl


protected theorem eqOn : EqOn (Prod.fst ∘ e) proj e.source := fun _x hx => e.coe_fst hx


                                                             /-
                                                               B : Type u_1
                                                               F : Type u_2
                                                               Z : Type u_4
                                                               inst✝² : TopologicalSpace B
                                                               inst✝¹ : TopologicalSpace F
                                                               proj : Z → B
                                                               inst✝ : TopologicalSpace Z
                                                               e : Trivialization F proj
                                                               x : Z
                                                               ⊢ Iff (Membership.mem e.source x) (Membership.mem e.baseSet (proj x))
                                                             -/
theorem mem_source : x ∈ e.source ↔ proj x ∈ e.baseSet := by rw [e.source_eq, mem_preimage]
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem source_inter_preimage_target_inter (s : Set (B × F)) :
    e.source ∩ e ⁻¹' (e.target ∩ s) = e.source ∩ e ⁻¹' s :=
  e.toPartialHomeomorph.source_inter_preimage_target_inter s


@[simp, mfld_simps]
theorem coe_mk (e : PartialHomeomorph Z (B × F)) (i j k l m) (x : Z) :
    (Trivialization.mk e i j k l m : Trivialization F proj) x = e x :=
  rfl


theorem mem_target {x : B × F} : x ∈ e.target ↔ x.1 ∈ e.baseSet :=
  e.toPretrivialization.mem_target


theorem map_target {x : B × F} (hx : x ∈ e.target) : e.toPartialHomeomorph.symm x ∈ e.source :=
  e.toPartialHomeomorph.map_target hx


theorem proj_symm_apply {x : B × F} (hx : x ∈ e.target) :
    proj (e.toPartialHomeomorph.symm x) = x.1 :=
  e.toPretrivialization.proj_symm_apply hx


theorem proj_symm_apply' {b : B} {x : F} (hx : b ∈ e.baseSet) :
    proj (e.toPartialHomeomorph.symm (b, x)) = b :=
  e.toPretrivialization.proj_symm_apply' hx


theorem proj_surjOn_baseSet [Nonempty F] : Set.SurjOn proj e.source e.baseSet :=
  e.toPretrivialization.proj_surjOn_baseSet


theorem apply_symm_apply {x : B × F} (hx : x ∈ e.target) : e (e.toPartialHomeomorph.symm x) = x :=
  e.toPartialHomeomorph.right_inv hx


theorem apply_symm_apply' {b : B} {x : F} (hx : b ∈ e.baseSet) :
    e (e.toPartialHomeomorph.symm (b, x)) = (b, x) :=
  e.toPretrivialization.apply_symm_apply' hx


@[simp, mfld_simps]
theorem symm_apply_mk_proj (ex : x ∈ e.source) : e.toPartialHomeomorph.symm (proj x, (e x).2) = x :=
  e.toPretrivialization.symm_apply_mk_proj ex


theorem symm_trans_source_eq (e e' : Trivialization F proj) :
    (e.toPartialEquiv.symm.trans e'.toPartialEquiv).source = (e.baseSet ∩ e'.baseSet) ×ˢ univ :=
  Pretrivialization.symm_trans_source_eq e.toPretrivialization e'


theorem symm_trans_target_eq (e e' : Trivialization F proj) :
    (e.toPartialEquiv.symm.trans e'.toPartialEquiv).target = (e.baseSet ∩ e'.baseSet) ×ˢ univ :=
  Pretrivialization.symm_trans_target_eq e.toPretrivialization e'


theorem coe_fst_eventuallyEq_proj (ex : x ∈ e.source) : Prod.fst ∘ e =ᶠ[𝓝 x] proj :=
  mem_nhds_iff.2 ⟨e.source, fun _y hy => e.coe_fst hy, e.open_source, ex⟩


theorem coe_fst_eventuallyEq_proj' (ex : proj x ∈ e.baseSet) : Prod.fst ∘ e =ᶠ[𝓝 x] proj :=
  e.coe_fst_eventuallyEq_proj (e.mem_source.2 ex)


theorem map_proj_nhds (ex : x ∈ e.source) : map proj (𝓝 x) = 𝓝 (proj x) := by
  rw [← e.coe_fst ex, ← map_congr (e.coe_fst_eventuallyEq_proj ex), ← map_map, ← e.coe_coe,
    e.map_nhds_eq ex, map_fst_nhds]


theorem preimage_subset_source {s : Set B} (hb : s ⊆ e.baseSet) : proj ⁻¹' s ⊆ e.source :=
  fun _p hp => e.mem_source.mpr (hb hp)


theorem image_preimage_eq_prod_univ {s : Set B} (hb : s ⊆ e.baseSet) :
    e '' (proj ⁻¹' s) = s ×ˢ univ :=
  Subset.antisymm
    (image_subset_iff.mpr fun p hp =>
      ⟨(e.proj_toFun p (e.preimage_subset_source hb hp)).symm ▸ hp, trivial⟩)
    fun p hp =>
    let hp' : p ∈ e.target := e.mem_target.mpr (hb hp.1)
    ⟨e.invFun p, mem_preimage.mpr ((e.proj_symm_apply hp').symm ▸ hp.1), e.apply_symm_apply hp'⟩


theorem tendsto_nhds_iff {α : Type*} {l : Filter α} {f : α → Z} {z : Z} (hz : z ∈ e.source) :
    Tendsto f l (𝓝 z) ↔
      Tendsto (proj ∘ f) l (𝓝 (proj z)) ∧ Tendsto (fun x ↦ (e (f x)).2) l (𝓝 (e z).2) := by
  rw [e.nhds_eq_comap_inf_principal hz, tendsto_inf, tendsto_comap_iff, Prod.tendsto_iff, coe_coe,
    tendsto_principal, coe_fst _ hz]
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    proj : Z → B
    inst✝ : TopologicalSpace Z
    e : Trivialization F proj
    α : Type u_5
    l : Filter α
    f : α → Z
    z : Z
    hz : Membership.mem e.source z
    ⊢ Iff (And (And (Filter.Tendsto (fun n => (Function.comp (↑e) f n).1) l (nhds  …
  -/
  by_cases hl : ∀ᶠ x in l, f x ∈ e.source
    /-
      case pos
      B : Type u_1
      F : Type u_2
      Z : Type u_4
      inst✝² : TopologicalSpace B
      inst✝¹ : TopologicalSpace F
      proj : Z → B
      inst✝ : TopologicalSpace Z
      e : Trivialization F proj
      α : Type u_5
      l : Filter α
      f : α → Z
      z : Z
      hz : Membership.mem e.source z
      hl : Filter.Eventually (fun x => Membership.mem e.source (f x)) l
      ⊢ Iff (And (And (Filter.Tendsto (fun n => (Function.comp (↑e) f n).1) l (nhds  …
    -/
  · simp only [hl, and_true]
    /-
      case pos
      B : Type u_1
      F : Type u_2
      Z : Type u_4
      inst✝² : TopologicalSpace B
      inst✝¹ : TopologicalSpace F
      proj : Z → B
      inst✝ : TopologicalSpace Z
      e : Trivialization F proj
      α : Type u_5
      l : Filter α
      f : α → Z
      z : Z
      hz : Membership.mem e.source z
      hl : Filter.Eventually (fun x => Membership.mem e.source (f x)) l
      ⊢ Iff (And (Filter.Tendsto (fun n => (Function.comp (↑e) f n).1) l (nhds (proj …
    -/
    refine (tendsto_congr' ?_).and Iff.rfl
    /-
      case pos
      B : Type u_1
      F : Type u_2
      Z : Type u_4
      inst✝² : TopologicalSpace B
      inst✝¹ : TopologicalSpace F
      proj : Z → B
      inst✝ : TopologicalSpace Z
      e : Trivialization F proj
      α : Type u_5
      l : Filter α
      f : α → Z
      z : Z
      hz : Membership.mem e.source z
      hl : Filter.Eventually (fun x => Membership.mem e.source (f x)) l
      ⊢ l.EventuallyEq (fun n => (Function.comp (↑e) f n).1) (Function.comp proj f)
    -/
    exact hl.mono fun x ↦ e.coe_fst
    /-
      🎉 no goals
    -/
    /-
      case neg
      B : Type u_1
      F : Type u_2
      Z : Type u_4
      inst✝² : TopologicalSpace B
      inst✝¹ : TopologicalSpace F
      proj : Z → B
      inst✝ : TopologicalSpace Z
      e : Trivialization F proj
      α : Type u_5
      l : Filter α
      f : α → Z
      z : Z
      hz : Membership.mem e.source z
      hl : Not (Filter.Eventually (fun x => Membership.mem e.source (f x)) l)
      ⊢ Iff (And (And (Filter.Tendsto (fun n => (Function.comp (↑e) f n).1) l (nhds  …
    -/
  · simp only [hl, and_false, false_iff, not_and]
    /-
      case neg
      B : Type u_1
      F : Type u_2
      Z : Type u_4
      inst✝² : TopologicalSpace B
      inst✝¹ : TopologicalSpace F
      proj : Z → B
      inst✝ : TopologicalSpace Z
      e : Trivialization F proj
      α : Type u_5
      l : Filter α
      f : α → Z
      z : Z
      hz : Membership.mem e.source z
      hl : Not (Filter.Eventually (fun x => Membership.mem e.source (f x)) l)
      ⊢ Filter.Tendsto (Function.comp proj f) l (nhds (proj z)) → Not (Filter.Tendst …
    -/
    rw [e.source_eq] at hl hz
    /-
      case neg
      B : Type u_1
      F : Type u_2
      Z : Type u_4
      inst✝² : TopologicalSpace B
      inst✝¹ : TopologicalSpace F
      proj : Z → B
      inst✝ : TopologicalSpace Z
      e : Trivialization F proj
      α : Type u_5
      l : Filter α
      f : α → Z
      z : Z
      hz : Membership.mem (Set.preimage proj e.baseSet) z
      hl : Not (Filter.Eventually (fun x => Membership.mem (Set.preimage proj e.base …
      ⊢ Filter.Tendsto (Function.comp proj f) l (nhds (proj z)) → Not (Filter.Tendst …
    -/
    exact fun h _ ↦ hl <| h <| e.open_baseSet.mem_nhds hz
    /-
      🎉 no goals
    -/


theorem nhds_eq_inf_comap {z : Z} (hz : z ∈ e.source) :
    𝓝 z = comap proj (𝓝 (proj z)) ⊓ comap (Prod.snd ∘ e) (𝓝 (e z).2) := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    proj : Z → B
    inst✝ : TopologicalSpace Z
    e : Trivialization F proj
    z : Z
    hz : Membership.mem e.source z
    ⊢ Eq (nhds z) (Min.min (Filter.comap proj (nhds (proj z))) (Filter.comap (Func …
  -/
  refine eq_of_forall_le_iff fun l ↦ ?_
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    proj : Z → B
    inst✝ : TopologicalSpace Z
    e : Trivialization F proj
    z : Z
    hz : Membership.mem e.source z
    l : Filter Z
    ⊢ Iff (LE.le l (nhds z)) (LE.le l (Min.min (Filter.comap proj (nhds (proj z))) …
  -/
  rw [le_inf_iff, ← tendsto_iff_comap, ← tendsto_iff_comap]
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    proj : Z → B
    inst✝ : TopologicalSpace Z
    e : Trivialization F proj
    z : Z
    hz : Membership.mem e.source z
    l : Filter Z
    ⊢ Iff (LE.le l (nhds z)) (And (Filter.Tendsto proj l (nhds (proj z))) (Filter. …
  -/
  exact e.tendsto_nhds_iff hz
  /-
    🎉 no goals
  -/


/-- The preimage of a subset of the base set is homeomorphic to the product with the fiber. -/
def preimageHomeomorph {s : Set B} (hb : s ⊆ e.baseSet) : proj ⁻¹' s ≃ₜ s × F :=
  (e.toPartialHomeomorph.homeomorphOfImageSubsetSource (e.preimage_subset_source hb)
        (e.image_preimage_eq_prod_univ hb)).trans
    ((Homeomorph.Set.prod s univ).trans ((Homeomorph.refl s).prodCongr (Homeomorph.Set.univ F)))


@[simp]
theorem preimageHomeomorph_apply {s : Set B} (hb : s ⊆ e.baseSet) (p : proj ⁻¹' s) :
    e.preimageHomeomorph hb p = (⟨proj p, p.2⟩, (e p).2) :=
  Prod.ext (Subtype.ext (e.proj_toFun p (e.mem_source.mpr (hb p.2)))) rfl


/-- Auxiliary definition to avoid looping in `dsimp`
with `Trivialization.preimageHomeomorph_symm_apply`. -/
protected def preimageHomeomorph_symm_apply.aux {s : Set B} (hb : s ⊆ e.baseSet) :=
  (e.preimageHomeomorph hb).symm


@[simp]
theorem preimageHomeomorph_symm_apply {s : Set B} (hb : s ⊆ e.baseSet) (p : s × F) :
    (e.preimageHomeomorph hb).symm p =
      ⟨e.symm (p.1, p.2), ((preimageHomeomorph_symm_apply.aux e hb) p).2⟩ :=
  rfl


/-- The source is homeomorphic to the product of the base set with the fiber. -/
def sourceHomeomorphBaseSetProd : e.source ≃ₜ e.baseSet × F :=
  (Homeomorph.setCongr e.source_eq).trans (e.preimageHomeomorph subset_rfl)


@[simp]
theorem sourceHomeomorphBaseSetProd_apply (p : e.source) :
    e.sourceHomeomorphBaseSetProd p = (⟨proj p, e.mem_source.mp p.2⟩, (e p).2) :=
  e.preimageHomeomorph_apply subset_rfl ⟨p, e.mem_source.mp p.2⟩


/-- Auxiliary definition to avoid looping in `dsimp`
with `Trivialization.sourceHomeomorphBaseSetProd_symm_apply`. -/
protected def sourceHomeomorphBaseSetProd_symm_apply.aux := e.sourceHomeomorphBaseSetProd.symm


@[simp]
theorem sourceHomeomorphBaseSetProd_symm_apply (p : e.baseSet × F) :
    e.sourceHomeomorphBaseSetProd.symm p =
      ⟨e.symm (p.1, p.2), (sourceHomeomorphBaseSetProd_symm_apply.aux e p).2⟩ :=
  rfl


/-- Each fiber of a trivialization is homeomorphic to the specified fiber. -/
def preimageSingletonHomeomorph {b : B} (hb : b ∈ e.baseSet) : proj ⁻¹' {b} ≃ₜ F :=
  .trans (e.preimageHomeomorph (Set.singleton_subset_iff.mpr hb)) <|
    .trans (.prodCongr (Homeomorph.homeomorphOfUnique ({b} : Set B) PUnit.{1}) (Homeomorph.refl F))
      (Homeomorph.punitProd F)


@[simp]
theorem preimageSingletonHomeomorph_apply {b : B} (hb : b ∈ e.baseSet) (p : proj ⁻¹' {b}) :
    e.preimageSingletonHomeomorph hb p = (e p).2 :=
  rfl


@[simp]
theorem preimageSingletonHomeomorph_symm_apply {b : B} (hb : b ∈ e.baseSet) (p : F) :
    (e.preimageSingletonHomeomorph hb).symm p =
                         /-
                           B : Type u_1
                           F : Type u_2
                           E : B → Type u_3
                           Z : Type u_4
                           inst✝³ : TopologicalSpace B
                           inst✝² : TopologicalSpace F
                           proj : Z → B
                           inst✝¹ : TopologicalSpace Z
                           inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
                           e : Trivialization F proj
                           x : Z
                           b : B
                           hb : Membership.mem e.baseSet b
                           p : F
                           ⊢ Membership.mem (Set.preimage proj (Singleton.singleton b)) (↑e.symm { fst := …
                         -/
      ⟨e.symm (b, p), by rw [mem_preimage, e.proj_symm_apply' hb, mem_singleton_iff]⟩ :=
                         /-
                           🎉 no goals
                         -/
  rfl


/-- In the domain of a bundle trivialization, the projection is continuous -/
theorem continuousAt_proj (ex : x ∈ e.source) : ContinuousAt proj x :=
  (e.map_proj_nhds ex).le


/-- Composition of a `Trivialization` and a `Homeomorph`. -/
protected def compHomeomorph {Z' : Type*} [TopologicalSpace Z'] (h : Z' ≃ₜ Z) :
    Trivialization F (proj ∘ h) where
  toPartialHomeomorph := h.toPartialHomeomorph.trans e.toPartialHomeomorph
  baseSet := e.baseSet
  open_baseSet := e.open_baseSet
                  /-
                    B : Type u_1
                    F : Type u_2
                    E : B → Type u_3
                    Z : Type u_4
                    inst✝⁴ : TopologicalSpace B
                    inst✝³ : TopologicalSpace F
                    proj : Z → B
                    inst✝² : TopologicalSpace Z
                    inst✝¹ : TopologicalSpace (Bundle.TotalSpace F E)
                    e : Trivialization F proj
                    x : Z
                    Z' : Type u_5
                    inst✝ : TopologicalSpace Z'
                    h : Homeomorph Z' Z
                    ⊢ Eq (h.toPartialHomeomorph.trans e.toPartialHomeomorph).source (Set.preimage  …
                  -/
  source_eq := by simp [source_eq, preimage_preimage, Function.comp_def]
                  /-
                    🎉 no goals
                  -/
                  /-
                    B : Type u_1
                    F : Type u_2
                    E : B → Type u_3
                    Z : Type u_4
                    inst✝⁴ : TopologicalSpace B
                    inst✝³ : TopologicalSpace F
                    proj : Z → B
                    inst✝² : TopologicalSpace Z
                    inst✝¹ : TopologicalSpace (Bundle.TotalSpace F E)
                    e : Trivialization F proj
                    x : Z
                    Z' : Type u_5
                    inst✝ : TopologicalSpace Z'
                    h : Homeomorph Z' Z
                    ⊢ Eq (h.toPartialHomeomorph.trans e.toPartialHomeomorph).target (SProd.sprod e …
                  -/
  target_eq := by simp [target_eq]
                  /-
                    🎉 no goals
                  -/
  proj_toFun p hp := by
    /-
      B : Type u_1
      F : Type u_2
      E : B → Type u_3
      Z : Type u_4
      inst✝⁴ : TopologicalSpace B
      inst✝³ : TopologicalSpace F
      proj : Z → B
      inst✝² : TopologicalSpace Z
      inst✝¹ : TopologicalSpace (Bundle.TotalSpace F E)
      e : Trivialization F proj
      x : Z
      Z' : Type u_5
      inst✝ : TopologicalSpace Z'
      h : Homeomorph Z' Z
      p : Z'
      hp : Membership.mem (h.toPartialHomeomorph.trans e.toPartialHomeomorph).source p
      ⊢ Eq (↑(h.toPartialHomeomorph.trans e.toPartialHomeomorph) p).1 (Function.comp …
    -/
    have hp : h p ∈ e.source := by simpa using hp
    /-
      B : Type u_1
      F : Type u_2
      E : B → Type u_3
      Z : Type u_4
      inst✝⁴ : TopologicalSpace B
      inst✝³ : TopologicalSpace F
      proj : Z → B
      inst✝² : TopologicalSpace Z
      inst✝¹ : TopologicalSpace (Bundle.TotalSpace F E)
      e : Trivialization F proj
      x : Z
      Z' : Type u_5
      inst✝ : TopologicalSpace Z'
      h : Homeomorph Z' Z
      p : Z'
      hp✝ : Membership.mem (h.toPartialHomeomorph.trans e.toPartialHomeomorph).sourc …
      hp : Membership.mem e.source (h p)
      ⊢ Eq (↑(h.toPartialHomeomorph.trans e.toPartialHomeomorph) p).1 (Function.comp …
    -/
    simp [hp]
    /-
      🎉 no goals
    -/


/-- Read off the continuity of a function `f : Z → X` at `z : Z` by transferring via a
trivialization of `Z` containing `z`. -/
theorem continuousAt_of_comp_right {X : Type*} [TopologicalSpace X] {f : Z → X} {z : Z}
    (e : Trivialization F proj) (he : proj z ∈ e.baseSet)
    (hf : ContinuousAt (f ∘ e.toPartialEquiv.symm) (e z)) : ContinuousAt f z := by
  have hez : z ∈ e.toPartialEquiv.symm.target := by
    rw [PartialEquiv.symm_target, e.mem_source]
    exact he
  rwa [e.toPartialHomeomorph.symm.continuousAt_iff_continuousAt_comp_right hez,
    PartialHomeomorph.symm_symm]


/-- Read off the continuity of a function `f : X → Z` at `x : X` by transferring via a
trivialization of `Z` containing `f x`. -/
theorem continuousAt_of_comp_left {X : Type*} [TopologicalSpace X] {f : X → Z} {x : X}
    (e : Trivialization F proj) (hf_proj : ContinuousAt (proj ∘ f) x) (he : proj (f x) ∈ e.baseSet)
    (hf : ContinuousAt (e ∘ f) x) : ContinuousAt f x := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝³ : TopologicalSpace B
    inst✝² : TopologicalSpace F
    proj : Z → B
    inst✝¹ : TopologicalSpace Z
    X : Type u_5
    inst✝ : TopologicalSpace X
    f : X → Z
    x : X
    e : Trivialization F proj
    hf_proj : ContinuousAt (Function.comp proj f) x
    he : Membership.mem e.baseSet (proj (f x))
    hf : ContinuousAt (Function.comp (↑e) f) x
    ⊢ ContinuousAt f x
  -/
  rw [e.continuousAt_iff_continuousAt_comp_left]
    /-
      B : Type u_1
      F : Type u_2
      Z : Type u_4
      inst✝³ : TopologicalSpace B
      inst✝² : TopologicalSpace F
      proj : Z → B
      inst✝¹ : TopologicalSpace Z
      X : Type u_5
      inst✝ : TopologicalSpace X
      f : X → Z
      x : X
      e : Trivialization F proj
      hf_proj : ContinuousAt (Function.comp proj f) x
      he : Membership.mem e.baseSet (proj (f x))
      hf : ContinuousAt (Function.comp (↑e) f) x
      ⊢ ContinuousAt (Function.comp (↑e.toPartialHomeomorph) f) x
    -/
  · exact hf
    /-
      🎉 no goals
    -/
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝³ : TopologicalSpace B
    inst✝² : TopologicalSpace F
    proj : Z → B
    inst✝¹ : TopologicalSpace Z
    X : Type u_5
    inst✝ : TopologicalSpace X
    f : X → Z
    x : X
    e : Trivialization F proj
    hf_proj : ContinuousAt (Function.comp proj f) x
    he : Membership.mem e.baseSet (proj (f x))
    hf : ContinuousAt (Function.comp (↑e) f) x
    ⊢ Membership.mem (nhds x) (Set.preimage f e.source)
  -/
  rw [e.source_eq, ← preimage_comp]
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝³ : TopologicalSpace B
    inst✝² : TopologicalSpace F
    proj : Z → B
    inst✝¹ : TopologicalSpace Z
    X : Type u_5
    inst✝ : TopologicalSpace X
    f : X → Z
    x : X
    e : Trivialization F proj
    hf_proj : ContinuousAt (Function.comp proj f) x
    he : Membership.mem e.baseSet (proj (f x))
    hf : ContinuousAt (Function.comp (↑e) f) x
    ⊢ Membership.mem (nhds x) (Set.preimage (Function.comp proj f) e.baseSet)
  -/
  exact hf_proj.preimage_mem_nhds (e.open_baseSet.mem_nhds he)
  /-
    🎉 no goals
  -/


protected theorem continuousOn : ContinuousOn e' e'.source :=
  e'.continuousOn_toFun


theorem coe_mem_source : ↑y ∈ e'.source ↔ b ∈ e'.baseSet :=
  e'.mem_source


theorem mk_mem_target {y : F} : (b, y) ∈ e'.target ↔ b ∈ e'.baseSet :=
  e'.toPretrivialization.mem_target


theorem symm_apply_apply {x : TotalSpace F E} (hx : x ∈ e'.source) :
    e'.toPartialHomeomorph.symm (e' x) = x :=
  e'.toPartialEquiv.left_inv hx


@[simp, mfld_simps]
theorem symm_coe_proj {x : B} {y : F} (e : Trivialization F (π F E)) (h : x ∈ e.baseSet) :
    (e.toPartialHomeomorph.symm (x, y)).1 = x :=
  e.proj_symm_apply' h


/-- A fiberwise inverse to `e'`. The function `F → E x` that induces a local inverse
`B × F → TotalSpace F E` of `e'` on `e'.baseSet`. It is defined to be `0` outside `e'.baseSet`. -/
protected noncomputable def symm (e : Trivialization F (π F E)) (b : B) (y : F) : E b :=
  e.toPretrivialization.symm b y


theorem symm_apply (e : Trivialization F (π F E)) {b : B} (hb : b ∈ e.baseSet) (y : F) :
    e.symm b y = cast (congr_arg E (e.symm_coe_proj hb)) (e.toPartialHomeomorph.symm (b, y)).2 :=
  dif_pos hb


theorem symm_apply_of_not_mem (e : Trivialization F (π F E)) {b : B} (hb : b ∉ e.baseSet) (y : F) :
    e.symm b y = 0 :=
  dif_neg hb


theorem mk_symm (e : Trivialization F (π F E)) {b : B} (hb : b ∈ e.baseSet) (y : F) :
    TotalSpace.mk b (e.symm b y) = e.toPartialHomeomorph.symm (b, y) :=
  e.toPretrivialization.mk_symm hb y


theorem symm_proj_apply (e : Trivialization F (π F E)) (z : TotalSpace F E)
    (hz : z.proj ∈ e.baseSet) : e.symm z.proj (e z).2 = z.2 :=
  e.toPretrivialization.symm_proj_apply z hz


theorem symm_apply_apply_mk (e : Trivialization F (π F E)) {b : B} (hb : b ∈ e.baseSet) (y : E b) :
    e.symm b (e ⟨b, y⟩).2 = y :=
  e.symm_proj_apply ⟨b, y⟩ hb


theorem apply_mk_symm (e : Trivialization F (π F E)) {b : B} (hb : b ∈ e.baseSet) (y : F) :
    e ⟨b, e.symm b y⟩ = (b, y) :=
  e.toPretrivialization.apply_mk_symm hb y


theorem continuousOn_symm (e : Trivialization F (π F E)) :
    ContinuousOn (fun z : B × F => TotalSpace.mk' F z.1 (e.symm z.1 z.2)) (e.baseSet ×ˢ univ) := by
  have : ∀ z ∈ e.baseSet ×ˢ (univ : Set F),
      TotalSpace.mk z.1 (e.symm z.1 z.2) = e.toPartialHomeomorph.symm z := by
    rintro x ⟨hx : x.1 ∈ e.baseSet, _⟩
    rw [e.mk_symm hx]
  /-
    B : Type u_1
    F : Type u_2
    E : B → Type u_3
    inst✝³ : TopologicalSpace B
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝ : (x : B) → Zero (E x)
    e : Trivialization F Bundle.TotalSpace.proj
    this : ∀ (z : Prod B F), Membership.mem (SProd.sprod e.baseSet Set.univ) z → E …
    ⊢ ContinuousOn (fun z => Bundle.TotalSpace.mk' F z.1 (e.symm z.1 z.2)) (SProd. …
  -/
  refine ContinuousOn.congr ?_ this
  /-
    B : Type u_1
    F : Type u_2
    E : B → Type u_3
    inst✝³ : TopologicalSpace B
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝ : (x : B) → Zero (E x)
    e : Trivialization F Bundle.TotalSpace.proj
    this : ∀ (z : Prod B F), Membership.mem (SProd.sprod e.baseSet Set.univ) z → E …
    ⊢ ContinuousOn (↑e.symm) (SProd.sprod e.baseSet Set.univ)
  -/
  rw [← e.target_eq]
  /-
    B : Type u_1
    F : Type u_2
    E : B → Type u_3
    inst✝³ : TopologicalSpace B
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalSpace (Bundle.TotalSpace F E)
    inst✝ : (x : B) → Zero (E x)
    e : Trivialization F Bundle.TotalSpace.proj
    this : ∀ (z : Prod B F), Membership.mem (SProd.sprod e.baseSet Set.univ) z → E …
    ⊢ ContinuousOn (↑e.symm) e.target
  -/
  exact e.toPartialHomeomorph.continuousOn_symm
  /-
    🎉 no goals
  -/


/-- If `e` is a `Trivialization` of `proj : Z → B` with fiber `F` and `h` is a homeomorphism
`F ≃ₜ F'`, then `e.trans_fiber_homeomorph h` is the trivialization of `proj` with the fiber `F'`
that sends `p : Z` to `((e p).1, h (e p).2)`. -/
def transFiberHomeomorph {F' : Type*} [TopologicalSpace F'] (e : Trivialization F proj)
    (h : F ≃ₜ F') : Trivialization F' proj where
  toPartialHomeomorph := e.toPartialHomeomorph.transHomeomorph <| (Homeomorph.refl _).prodCongr h
  baseSet := e.baseSet
  open_baseSet := e.open_baseSet
  source_eq := e.source_eq
                  /-
                    B : Type u_1
                    F : Type u_2
                    E : B → Type u_3
                    Z : Type u_4
                    inst✝⁴ : TopologicalSpace B
                    inst✝³ : TopologicalSpace F
                    proj : Z → B
                    inst✝² : TopologicalSpace Z
                    inst✝¹ : TopologicalSpace (Bundle.TotalSpace F E)
                    e✝ : Trivialization F proj
                    x : Z
                    e' : Trivialization F Bundle.TotalSpace.proj
                    b : B
                    y : E b
                    F' : Type u_5
                    inst✝ : TopologicalSpace F'
                    e : Trivialization F proj
                    h : Homeomorph F F'
                    ⊢ Eq (e.transHomeomorph ((Homeomorph.refl B).prodCongr h)).target (SProd.sprod …
                  -/
  target_eq := by simp [target_eq, prod_univ, preimage_preimage]
                  /-
                    🎉 no goals
                  -/
  proj_toFun := e.proj_toFun


@[simp]
theorem transFiberHomeomorph_apply {F' : Type*} [TopologicalSpace F'] (e : Trivialization F proj)
    (h : F ≃ₜ F') (x : Z) : e.transFiberHomeomorph h x = ((e x).1, h (e x).2) :=
  rfl


/-- Coordinate transformation in the fiber induced by a pair of bundle trivializations. See also
`Trivialization.coordChangeHomeomorph` for a version bundled as `F ≃ₜ F`. -/
def coordChange (e₁ e₂ : Trivialization F proj) (b : B) (x : F) : F :=
  (e₂ <| e₁.toPartialHomeomorph.symm (b, x)).2


theorem mk_coordChange (e₁ e₂ : Trivialization F proj) {b : B} (h₁ : b ∈ e₁.baseSet)
    (h₂ : b ∈ e₂.baseSet) (x : F) :
    (b, e₁.coordChange e₂ b x) = e₂ (e₁.toPartialHomeomorph.symm (b, x)) := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    proj : Z → B
    inst✝ : TopologicalSpace Z
    e₁ e₂ : Trivialization F proj
    b : B
    h₁ : Membership.mem e₁.baseSet b
    h₂ : Membership.mem e₂.baseSet b
    x : F
    ⊢ Eq { fst := b, snd := e₁.coordChange e₂ b x } (↑e₂ (↑e₁.symm { fst := b, snd …
  -/
  refine Prod.ext ?_ rfl
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    proj : Z → B
    inst✝ : TopologicalSpace Z
    e₁ e₂ : Trivialization F proj
    b : B
    h₁ : Membership.mem e₁.baseSet b
    h₂ : Membership.mem e₂.baseSet b
    x : F
    ⊢ Eq { fst := b, snd := e₁.coordChange e₂ b x }.1 (↑e₂ (↑e₁.symm { fst := b, s …
  -/
  rw [e₂.coe_fst', ← e₁.coe_fst', e₁.apply_symm_apply' h₁]
    /-
      B : Type u_1
      F : Type u_2
      Z : Type u_4
      inst✝² : TopologicalSpace B
      inst✝¹ : TopologicalSpace F
      proj : Z → B
      inst✝ : TopologicalSpace Z
      e₁ e₂ : Trivialization F proj
      b : B
      h₁ : Membership.mem e₁.baseSet b
      h₂ : Membership.mem e₂.baseSet b
      x : F
      ⊢ Membership.mem e₁.baseSet (proj (↑e₁.symm { fst := b, snd := x }))
    -/
  · rwa [e₁.proj_symm_apply' h₁]
    /-
      🎉 no goals
    -/
    /-
      B : Type u_1
      F : Type u_2
      Z : Type u_4
      inst✝² : TopologicalSpace B
      inst✝¹ : TopologicalSpace F
      proj : Z → B
      inst✝ : TopologicalSpace Z
      e₁ e₂ : Trivialization F proj
      b : B
      h₁ : Membership.mem e₁.baseSet b
      h₂ : Membership.mem e₂.baseSet b
      x : F
      ⊢ Membership.mem e₂.baseSet (proj (↑e₁.symm { fst := b, snd := x }))
    -/
  · rwa [e₁.proj_symm_apply' h₁]
    /-
      🎉 no goals
    -/


theorem coordChange_apply_snd (e₁ e₂ : Trivialization F proj) {p : Z} (h : proj p ∈ e₁.baseSet) :
    e₁.coordChange e₂ (proj p) (e₁ p).snd = (e₂ p).snd := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    proj : Z → B
    inst✝ : TopologicalSpace Z
    e₁ e₂ : Trivialization F proj
    p : Z
    h : Membership.mem e₁.baseSet (proj p)
    ⊢ Eq (e₁.coordChange e₂ (proj p) (↑e₁ p).2) (↑e₂ p).2
  -/
  rw [coordChange, e₁.symm_apply_mk_proj (e₁.mem_source.2 h)]
  /-
    🎉 no goals
  -/


theorem coordChange_same_apply (e : Trivialization F proj) {b : B} (h : b ∈ e.baseSet) (x : F) :
                                  /-
                                    B : Type u_1
                                    F : Type u_2
                                    Z : Type u_4
                                    inst✝² : TopologicalSpace B
                                    inst✝¹ : TopologicalSpace F
                                    proj : Z → B
                                    inst✝ : TopologicalSpace Z
                                    e : Trivialization F proj
                                    b : B
                                    h : Membership.mem e.baseSet b
                                    x : F
                                    ⊢ Eq (e.coordChange e b x) x
                                  -/
    e.coordChange e b x = x := by rw [coordChange, e.apply_symm_apply' h]
                                  /-
                                    🎉 no goals
                                  -/


theorem coordChange_same (e : Trivialization F proj) {b : B} (h : b ∈ e.baseSet) :
    e.coordChange e b = id :=
  funext <| e.coordChange_same_apply h


theorem coordChange_coordChange (e₁ e₂ e₃ : Trivialization F proj) {b : B} (h₁ : b ∈ e₁.baseSet)
    (h₂ : b ∈ e₂.baseSet) (x : F) :
    e₂.coordChange e₃ b (e₁.coordChange e₂ b x) = e₁.coordChange e₃ b x := by
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    proj : Z → B
    inst✝ : TopologicalSpace Z
    e₁ e₂ e₃ : Trivialization F proj
    b : B
    h₁ : Membership.mem e₁.baseSet b
    h₂ : Membership.mem e₂.baseSet b
    x : F
    ⊢ Eq (e₂.coordChange e₃ b (e₁.coordChange e₂ b x)) (e₁.coordChange e₃ b x)
  -/
  rw [coordChange, e₁.mk_coordChange _ h₁ h₂, ← e₂.coe_coe, e₂.left_inv, coordChange]
  /-
    B : Type u_1
    F : Type u_2
    Z : Type u_4
    inst✝² : TopologicalSpace B
    inst✝¹ : TopologicalSpace F
    proj : Z → B
    inst✝ : TopologicalSpace Z
    e₁ e₂ e₃ : Trivialization F proj
    b : B
    h₁ : Membership.mem e₁.baseSet b
    h₂ : Membership.mem e₂.baseSet b
    x : F
    ⊢ Membership.mem e₂.source (↑e₁.symm { fst := b, snd := x })
  -/
  rwa [e₂.mem_source, e₁.proj_symm_apply' h₁]
  /-
    🎉 no goals
  -/


theorem continuous_coordChange (e₁ e₂ : Trivialization F proj) {b : B} (h₁ : b ∈ e₁.baseSet)
    (h₂ : b ∈ e₂.baseSet) : Continuous (e₁.coordChange e₂ b) := by
  refine continuous_snd.comp (e₂.toPartialHomeomorph.continuousOn.comp_continuous
    (e₁.toPartialHomeomorph.continuousOn_symm.comp_continuous ?_ ?_) ?_)
    /-
      case refine_1
      B : Type u_1
      F : Type u_2
      Z : Type u_4
      inst✝² : TopologicalSpace B
      inst✝¹ : TopologicalSpace F
      proj : Z → B
      inst✝ : TopologicalSpace Z
      e₁ e₂ : Trivialization F proj
      b : B
      h₁ : Membership.mem e₁.baseSet b
      h₂ : Membership.mem e₂.baseSet b
      ⊢ Continuous (Prod.mk b)
    -/
  · exact continuous_const.prod_mk continuous_id
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      B : Type u_1
      F : Type u_2
      Z : Type u_4
      inst✝² : TopologicalSpace B
      inst✝¹ : TopologicalSpace F
      proj : Z → B
      inst✝ : TopologicalSpace Z
      e₁ e₂ : Trivialization F proj
      b : B
      h₁ : Membership.mem e₁.baseSet b
      h₂ : Membership.mem e₂.baseSet b
      ⊢ ∀ (x : F), Membership.mem e₁.target { fst := b, snd := x }
    -/
  · exact fun x => e₁.mem_target.2 h₁
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      B : Type u_1
      F : Type u_2
      Z : Type u_4
      inst✝² : TopologicalSpace B
      inst✝¹ : TopologicalSpace F
      proj : Z → B
      inst✝ : TopologicalSpace Z
      e₁ e₂ : Trivialization F proj
      b : B
      h₁ : Membership.mem e₁.baseSet b
      h₂ : Membership.mem e₂.baseSet b
      ⊢ ∀ (x : F), Membership.mem e₂.source (↑e₁.symm { fst := b, snd := x })
    -/
  · intro x
    /-
      case refine_3
      B : Type u_1
      F : Type u_2
      Z : Type u_4
      inst✝² : TopologicalSpace B
      inst✝¹ : TopologicalSpace F
      proj : Z → B
      inst✝ : TopologicalSpace Z
      e₁ e₂ : Trivialization F proj
      b : B
      h₁ : Membership.mem e₁.baseSet b
      h₂ : Membership.mem e₂.baseSet b
      x : F
      ⊢ Membership.mem e₂.source (↑e₁.symm { fst := b, snd := x })
    -/
    rwa [e₂.mem_source, e₁.proj_symm_apply' h₁]
    /-
      🎉 no goals
    -/


/-- Coordinate transformation in the fiber induced by a pair of bundle trivializations,
as a homeomorphism. -/
protected def coordChangeHomeomorph (e₁ e₂ : Trivialization F proj) {b : B} (h₁ : b ∈ e₁.baseSet)
    (h₂ : b ∈ e₂.baseSet) : F ≃ₜ F where
  toFun := e₁.coordChange e₂ b
  invFun := e₂.coordChange e₁ b
                   /-
                     B : Type u_1
                     F : Type u_2
                     E : B → Type u_3
                     Z : Type u_4
                     inst✝³ : TopologicalSpace B
                     inst✝² : TopologicalSpace F
                     proj : Z → B
                     inst✝¹ : TopologicalSpace Z
                     inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
                     e : Trivialization F proj
                     x✝ : Z
                     e' : Trivialization F Bundle.TotalSpace.proj
                     b✝ : B
                     y : E b✝
                     e₁ e₂ : Trivialization F proj
                     b : B
                     h₁ : Membership.mem e₁.baseSet b
                     h₂ : Membership.mem e₂.baseSet b
                     x : F
                     ⊢ Eq (e₂.coordChange e₁ b (e₁.coordChange e₂ b x)) x
                   -/
  left_inv x := by simp only [*, coordChange_coordChange, coordChange_same_apply]
                   /-
                     🎉 no goals
                   -/
                    /-
                      B : Type u_1
                      F : Type u_2
                      E : B → Type u_3
                      Z : Type u_4
                      inst✝³ : TopologicalSpace B
                      inst✝² : TopologicalSpace F
                      proj : Z → B
                      inst✝¹ : TopologicalSpace Z
                      inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
                      e : Trivialization F proj
                      x✝ : Z
                      e' : Trivialization F Bundle.TotalSpace.proj
                      b✝ : B
                      y : E b✝
                      e₁ e₂ : Trivialization F proj
                      b : B
                      h₁ : Membership.mem e₁.baseSet b
                      h₂ : Membership.mem e₂.baseSet b
                      x : F
                      ⊢ Eq (e₁.coordChange e₂ b (e₂.coordChange e₁ b x)) x
                    -/
  right_inv x := by simp only [*, coordChange_coordChange, coordChange_same_apply]
                    /-
                      🎉 no goals
                    -/
  continuous_toFun := e₁.continuous_coordChange e₂ h₁ h₂
  continuous_invFun := e₂.continuous_coordChange e₁ h₂ h₁


@[simp]
theorem coordChangeHomeomorph_coe (e₁ e₂ : Trivialization F proj) {b : B} (h₁ : b ∈ e₁.baseSet)
    (h₂ : b ∈ e₂.baseSet) : ⇑(e₁.coordChangeHomeomorph e₂ h₁ h₂) = e₁.coordChange e₂ b :=
  rfl


theorem isImage_preimage_prod (e : Trivialization F proj) (s : Set B) :
                                                                             /-
                                                                               B : Type u_1
                                                                               F : Type u_2
                                                                               Z : Type u_4
                                                                               inst✝² : TopologicalSpace B
                                                                               inst✝¹ : TopologicalSpace F
                                                                               proj : Z → B
                                                                               inst✝ : TopologicalSpace Z
                                                                               e : Trivialization F proj
                                                                               s : Set B
                                                                               x : Z
                                                                               hx : Membership.mem e.source x
                                                                               ⊢ Iff (Membership.mem (SProd.sprod s Set.univ) (↑e.toPartialHomeomorph x)) (Me …
                                                                             -/
    e.toPartialHomeomorph.IsImage (proj ⁻¹' s) (s ×ˢ univ) := fun x hx => by simp [e.coe_fst', hx]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- Restrict a `Trivialization` to an open set in the base. -/
protected def restrOpen (e : Trivialization F proj) (s : Set B) (hs : IsOpen s) :
    Trivialization F proj where
  toPartialHomeomorph :=
    ((e.isImage_preimage_prod s).symm.restr (IsOpen.inter e.open_target (hs.prod isOpen_univ))).symm
  baseSet := e.baseSet ∩ s
  open_baseSet := IsOpen.inter e.open_baseSet hs
                  /-
                    B : Type u_1
                    F : Type u_2
                    E : B → Type u_3
                    Z : Type u_4
                    inst✝³ : TopologicalSpace B
                    inst✝² : TopologicalSpace F
                    proj : Z → B
                    inst✝¹ : TopologicalSpace Z
                    inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
                    e✝ : Trivialization F proj
                    x : Z
                    e' : Trivialization F Bundle.TotalSpace.proj
                    b : B
                    y : E b
                    e : Trivialization F proj
                    s : Set B
                    hs : IsOpen s
                    ⊢ Eq (⋯.restr ⋯).symm.source (Set.preimage proj (Inter.inter e.baseSet s))
                  -/
  source_eq := by simp [source_eq]
                  /-
                    🎉 no goals
                  -/
                  /-
                    B : Type u_1
                    F : Type u_2
                    E : B → Type u_3
                    Z : Type u_4
                    inst✝³ : TopologicalSpace B
                    inst✝² : TopologicalSpace F
                    proj : Z → B
                    inst✝¹ : TopologicalSpace Z
                    inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
                    e✝ : Trivialization F proj
                    x : Z
                    e' : Trivialization F Bundle.TotalSpace.proj
                    b : B
                    y : E b
                    e : Trivialization F proj
                    s : Set B
                    hs : IsOpen s
                    ⊢ Eq (⋯.restr ⋯).symm.target (SProd.sprod (Inter.inter e.baseSet s) Set.univ)
                  -/
  target_eq := by simp [target_eq, prod_univ]
                  /-
                    🎉 no goals
                  -/
  proj_toFun p hp := e.proj_toFun p hp.1


theorem frontier_preimage (e : Trivialization F proj) (s : Set B) :
    e.source ∩ frontier (proj ⁻¹' s) = proj ⁻¹' (e.baseSet ∩ frontier s) := by
  rw [← (e.isImage_preimage_prod s).frontier.preimage_eq, frontier_prod_univ_eq,
    (e.isImage_preimage_prod _).preimage_eq, e.source_eq, preimage_inter]


open Classical in
/-- Given two bundle trivializations `e`, `e'` of `proj : Z → B` and a set `s : Set B` such that
the base sets of `e` and `e'` intersect `frontier s` on the same set and `e p = e' p` whenever
`proj p ∈ e.baseSet ∩ frontier s`, `e.piecewise e' s Hs Heq` is the bundle trivialization over
`Set.ite s e.baseSet e'.baseSet` that is equal to `e` on `proj ⁻¹ s` and is equal to `e'`
otherwise. -/
noncomputable def piecewise (e e' : Trivialization F proj) (s : Set B)
    (Hs : e.baseSet ∩ frontier s = e'.baseSet ∩ frontier s)
    (Heq : EqOn e e' <| proj ⁻¹' (e.baseSet ∩ frontier s)) : Trivialization F proj where
  toPartialHomeomorph :=
    e.toPartialHomeomorph.piecewise e'.toPartialHomeomorph (proj ⁻¹' s) (s ×ˢ univ)
      (e.isImage_preimage_prod s) (e'.isImage_preimage_prod s)
          /-
            B : Type u_1
            F : Type u_2
            E : B → Type u_3
            Z : Type u_4
            inst✝³ : TopologicalSpace B
            inst✝² : TopologicalSpace F
            proj : Z → B
            inst✝¹ : TopologicalSpace Z
            inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
            e✝ : Trivialization F proj
            x : Z
            e'✝ : Trivialization F Bundle.TotalSpace.proj
            b : B
            y : E b
            e e' : Trivialization F proj
            s : Set B
            Hs : Eq (Inter.inter e.baseSet (frontier s)) (Inter.inter e'.baseSet (frontier …
            Heq : Set.EqOn (↑e) (↑e') (Set.preimage proj (Inter.inter e.baseSet (frontier  …
            ⊢ Eq (Inter.inter e.source (frontier (Set.preimage proj s))) (Inter.inter e'.s …
          -/
          /-
            🎉 no goals
          -/
      (by rw [e.frontier_preimage, e'.frontier_preimage, Hs]) (by rwa [e.frontier_preimage])
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
  baseSet := s.ite e.baseSet e'.baseSet
  open_baseSet := e.open_baseSet.ite e'.open_baseSet Hs
                  /-
                    B : Type u_1
                    F : Type u_2
                    E : B → Type u_3
                    Z : Type u_4
                    inst✝³ : TopologicalSpace B
                    inst✝² : TopologicalSpace F
                    proj : Z → B
                    inst✝¹ : TopologicalSpace Z
                    inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
                    e✝ : Trivialization F proj
                    x : Z
                    e'✝ : Trivialization F Bundle.TotalSpace.proj
                    b : B
                    y : E b
                    e e' : Trivialization F proj
                    s : Set B
                    Hs : Eq (Inter.inter e.baseSet (frontier s)) (Inter.inter e'.baseSet (frontier …
                    Heq : Set.EqOn (↑e) (↑e') (Set.preimage proj (Inter.inter e.baseSet (frontier  …
                    ⊢ Eq (e.piecewise e'.toPartialHomeomorph (Set.preimage proj s) (SProd.sprod s  …
                  -/
  source_eq := by simp [source_eq]
                  /-
                    🎉 no goals
                  -/
                  /-
                    B : Type u_1
                    F : Type u_2
                    E : B → Type u_3
                    Z : Type u_4
                    inst✝³ : TopologicalSpace B
                    inst✝² : TopologicalSpace F
                    proj : Z → B
                    inst✝¹ : TopologicalSpace Z
                    inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
                    e✝ : Trivialization F proj
                    x : Z
                    e'✝ : Trivialization F Bundle.TotalSpace.proj
                    b : B
                    y : E b
                    e e' : Trivialization F proj
                    s : Set B
                    Hs : Eq (Inter.inter e.baseSet (frontier s)) (Inter.inter e'.baseSet (frontier …
                    Heq : Set.EqOn (↑e) (↑e') (Set.preimage proj (Inter.inter e.baseSet (frontier  …
                    ⊢ Eq (e.piecewise e'.toPartialHomeomorph (Set.preimage proj s) (SProd.sprod s  …
                  -/
  target_eq := by simp [target_eq, prod_univ]
                  /-
                    🎉 no goals
                  -/
  proj_toFun p := by
    /-
      B : Type u_1
      F : Type u_2
      E : B → Type u_3
      Z : Type u_4
      inst✝³ : TopologicalSpace B
      inst✝² : TopologicalSpace F
      proj : Z → B
      inst✝¹ : TopologicalSpace Z
      inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
      e✝ : Trivialization F proj
      x : Z
      e'✝ : Trivialization F Bundle.TotalSpace.proj
      b : B
      y : E b
      e e' : Trivialization F proj
      s : Set B
      Hs : Eq (Inter.inter e.baseSet (frontier s)) (Inter.inter e'.baseSet (frontier …
      Heq : Set.EqOn (↑e) (↑e') (Set.preimage proj (Inter.inter e.baseSet (frontier  …
      p : Z
      ⊢ Membership.mem (e.piecewise e'.toPartialHomeomorph (Set.preimage proj s) (SP …
    -/
                                     /-
                                       🎉 no goals
                                     -/
    rintro (⟨he, hs⟩ | ⟨he, hs⟩) <;> simp [*]
                                     /-
                                       🎉 no goals
                                     -/


/-- Given two bundle trivializations `e`, `e'` of a topological fiber bundle `proj : Z → B`
over a linearly ordered base `B` and a point `a ∈ e.baseSet ∩ e'.baseSet` such that
`e` equals `e'` on `proj ⁻¹' {a}`, `e.piecewise_le_of_eq e' a He He' Heq` is the bundle
trivialization over `Set.ite (Iic a) e.baseSet e'.baseSet` that is equal to `e` on points `p`
such that `proj p ≤ a` and is equal to `e'` otherwise. -/
noncomputable def piecewiseLeOfEq [LinearOrder B] [OrderTopology B] (e e' : Trivialization F proj)
    (a : B) (He : a ∈ e.baseSet) (He' : a ∈ e'.baseSet) (Heq : ∀ p, proj p = a → e p = e' p) :
    Trivialization F proj :=
  e.piecewise e' (Iic a)
    (Set.ext fun x => and_congr_left_iff.2 fun hx => by
      /-
        B : Type u_1
        F : Type u_2
        E : B → Type u_3
        Z : Type u_4
        inst✝⁵ : TopologicalSpace B
        inst✝⁴ : TopologicalSpace F
        proj : Z → B
        inst✝³ : TopologicalSpace Z
        inst✝² : TopologicalSpace (Bundle.TotalSpace F E)
        e✝ : Trivialization F proj
        x✝ : Z
        e'✝ : Trivialization F Bundle.TotalSpace.proj
        b : B
        y : E b
        inst✝¹ : LinearOrder B
        inst✝ : OrderTopology B
        e e' : Trivialization F proj
        a : B
        He : Membership.mem e.baseSet a
        He' : Membership.mem e'.baseSet a
        Heq : ∀ (p : Z), Eq (proj p) a → Eq (↑e p) (↑e' p)
        x : B
        hx : Membership.mem (frontier (Set.Iic a)) x
        ⊢ Iff (Membership.mem e.baseSet x) (Membership.mem e'.baseSet x)
      -/
      obtain rfl : x = a := mem_singleton_iff.1 (frontier_Iic_subset _ hx)
      /-
        B : Type u_1
        F : Type u_2
        E : B → Type u_3
        Z : Type u_4
        inst✝⁵ : TopologicalSpace B
        inst✝⁴ : TopologicalSpace F
        proj : Z → B
        inst✝³ : TopologicalSpace Z
        inst✝² : TopologicalSpace (Bundle.TotalSpace F E)
        e✝ : Trivialization F proj
        x✝ : Z
        e'✝ : Trivialization F Bundle.TotalSpace.proj
        b : B
        y : E b
        inst✝¹ : LinearOrder B
        inst✝ : OrderTopology B
        e e' : Trivialization F proj
        x : B
        He : Membership.mem e.baseSet x
        He' : Membership.mem e'.baseSet x
        Heq : ∀ (p : Z), Eq (proj p) x → Eq (↑e p) (↑e' p)
        hx : Membership.mem (frontier (Set.Iic x)) x
        ⊢ Iff (Membership.mem e.baseSet x) (Membership.mem e'.baseSet x)
      -/
      simp [He, He'])
      /-
        🎉 no goals
      -/
    fun p hp => Heq p <| frontier_Iic_subset _ hp.2


/-- Given two bundle trivializations `e`, `e'` of a topological fiber bundle `proj : Z → B` over a
linearly ordered base `B` and a point `a ∈ e.baseSet ∩ e'.baseSet`, `e.piecewise_le e' a He He'`
is the bundle trivialization over `Set.ite (Iic a) e.baseSet e'.baseSet` that is equal to `e` on
points `p` such that `proj p ≤ a` and is equal to `((e' p).1, h (e' p).2)` otherwise, where
`h = e'.coord_change_homeomorph e _ _` is the homeomorphism of the fiber such that
`h (e' p).2 = (e p).2` whenever `e p = a`. -/
noncomputable def piecewiseLe [LinearOrder B] [OrderTopology B] (e e' : Trivialization F proj)
    (a : B) (He : a ∈ e.baseSet) (He' : a ∈ e'.baseSet) : Trivialization F proj :=
  e.piecewiseLeOfEq (e'.transFiberHomeomorph (e'.coordChangeHomeomorph e He' He)) a He He' <| by
    /-
      B : Type u_1
      F : Type u_2
      E : B → Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace B
      inst✝⁴ : TopologicalSpace F
      proj : Z → B
      inst✝³ : TopologicalSpace Z
      inst✝² : TopologicalSpace (Bundle.TotalSpace F E)
      e✝ : Trivialization F proj
      x : Z
      e'✝ : Trivialization F Bundle.TotalSpace.proj
      b : B
      y : E b
      inst✝¹ : LinearOrder B
      inst✝ : OrderTopology B
      e e' : Trivialization F proj
      a : B
      He : Membership.mem e.baseSet a
      He' : Membership.mem e'.baseSet a
      ⊢ ∀ (p : Z), Eq (proj p) a → Eq (↑e p) (↑(e'.transFiberHomeomorph (e'.coordCha …
    -/
    rintro p rfl
    /-
      B : Type u_1
      F : Type u_2
      E : B → Type u_3
      Z : Type u_4
      inst✝⁵ : TopologicalSpace B
      inst✝⁴ : TopologicalSpace F
      proj : Z → B
      inst✝³ : TopologicalSpace Z
      inst✝² : TopologicalSpace (Bundle.TotalSpace F E)
      e✝ : Trivialization F proj
      x : Z
      e'✝ : Trivialization F Bundle.TotalSpace.proj
      b : B
      y : E b
      inst✝¹ : LinearOrder B
      inst✝ : OrderTopology B
      e e' : Trivialization F proj
      p : Z
      He : Membership.mem e.baseSet (proj p)
      He' : Membership.mem e'.baseSet (proj p)
      ⊢ Eq (↑e p) (↑(e'.transFiberHomeomorph (e'.coordChangeHomeomorph e He' He)) p)
    -/
    ext1
      /-
        case fst
        B : Type u_1
        F : Type u_2
        E : B → Type u_3
        Z : Type u_4
        inst✝⁵ : TopologicalSpace B
        inst✝⁴ : TopologicalSpace F
        proj : Z → B
        inst✝³ : TopologicalSpace Z
        inst✝² : TopologicalSpace (Bundle.TotalSpace F E)
        e✝ : Trivialization F proj
        x : Z
        e'✝ : Trivialization F Bundle.TotalSpace.proj
        b : B
        y : E b
        inst✝¹ : LinearOrder B
        inst✝ : OrderTopology B
        e e' : Trivialization F proj
        p : Z
        He : Membership.mem e.baseSet (proj p)
        He' : Membership.mem e'.baseSet (proj p)
        ⊢ Eq (↑e p).1 (↑(e'.transFiberHomeomorph (e'.coordChangeHomeomorph e He' He))  …
      -/
    · simp [e.coe_fst', e'.coe_fst', *]
      /-
        🎉 no goals
      -/
      /-
        case snd
        B : Type u_1
        F : Type u_2
        E : B → Type u_3
        Z : Type u_4
        inst✝⁵ : TopologicalSpace B
        inst✝⁴ : TopologicalSpace F
        proj : Z → B
        inst✝³ : TopologicalSpace Z
        inst✝² : TopologicalSpace (Bundle.TotalSpace F E)
        e✝ : Trivialization F proj
        x : Z
        e'✝ : Trivialization F Bundle.TotalSpace.proj
        b : B
        y : E b
        inst✝¹ : LinearOrder B
        inst✝ : OrderTopology B
        e e' : Trivialization F proj
        p : Z
        He : Membership.mem e.baseSet (proj p)
        He' : Membership.mem e'.baseSet (proj p)
        ⊢ Eq (↑e p).2 (↑(e'.transFiberHomeomorph (e'.coordChangeHomeomorph e He' He))  …
      -/
    · simp [coordChange_apply_snd, *]
      /-
        🎉 no goals
      -/


open Classical in
/-- Given two bundle trivializations `e`, `e'` over disjoint sets, `e.disjoint_union e' H` is the
bundle trivialization over the union of the base sets that agrees with `e` and `e'` over their
base sets. -/
noncomputable def disjointUnion (e e' : Trivialization F proj) (H : Disjoint e.baseSet e'.baseSet) :
    Trivialization F proj where
  toPartialHomeomorph :=
    e.toPartialHomeomorph.disjointUnion e'.toPartialHomeomorph
      (by
        /-
          B : Type u_1
          F : Type u_2
          E : B → Type u_3
          Z : Type u_4
          inst✝³ : TopologicalSpace B
          inst✝² : TopologicalSpace F
          proj : Z → B
          inst✝¹ : TopologicalSpace Z
          inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
          e✝ : Trivialization F proj
          x : Z
          e'✝ : Trivialization F Bundle.TotalSpace.proj
          b : B
          y : E b
          e e' : Trivialization F proj
          H : Disjoint e.baseSet e'.baseSet
          ⊢ Disjoint e.source e'.source
        -/
        rw [e.source_eq, e'.source_eq]
        /-
          B : Type u_1
          F : Type u_2
          E : B → Type u_3
          Z : Type u_4
          inst✝³ : TopologicalSpace B
          inst✝² : TopologicalSpace F
          proj : Z → B
          inst✝¹ : TopologicalSpace Z
          inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
          e✝ : Trivialization F proj
          x : Z
          e'✝ : Trivialization F Bundle.TotalSpace.proj
          b : B
          y : E b
          e e' : Trivialization F proj
          H : Disjoint e.baseSet e'.baseSet
          ⊢ Disjoint (Set.preimage proj e.baseSet) (Set.preimage proj e'.baseSet)
        -/
        exact H.preimage _)
        /-
          🎉 no goals
        -/
      (by
        /-
          B : Type u_1
          F : Type u_2
          E : B → Type u_3
          Z : Type u_4
          inst✝³ : TopologicalSpace B
          inst✝² : TopologicalSpace F
          proj : Z → B
          inst✝¹ : TopologicalSpace Z
          inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
          e✝ : Trivialization F proj
          x : Z
          e'✝ : Trivialization F Bundle.TotalSpace.proj
          b : B
          y : E b
          e e' : Trivialization F proj
          H : Disjoint e.baseSet e'.baseSet
          ⊢ Disjoint e.target e'.target
        -/
        rw [e.target_eq, e'.target_eq, disjoint_iff_inf_le]
        /-
          B : Type u_1
          F : Type u_2
          E : B → Type u_3
          Z : Type u_4
          inst✝³ : TopologicalSpace B
          inst✝² : TopologicalSpace F
          proj : Z → B
          inst✝¹ : TopologicalSpace Z
          inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
          e✝ : Trivialization F proj
          x : Z
          e'✝ : Trivialization F Bundle.TotalSpace.proj
          b : B
          y : E b
          e e' : Trivialization F proj
          H : Disjoint e.baseSet e'.baseSet
          ⊢ LE.le (Min.min (SProd.sprod e.baseSet Set.univ) (SProd.sprod e'.baseSet Set. …
        -/
        intro x hx
        /-
          B : Type u_1
          F : Type u_2
          E : B → Type u_3
          Z : Type u_4
          inst✝³ : TopologicalSpace B
          inst✝² : TopologicalSpace F
          proj : Z → B
          inst✝¹ : TopologicalSpace Z
          inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
          e✝ : Trivialization F proj
          x✝ : Z
          e'✝ : Trivialization F Bundle.TotalSpace.proj
          b : B
          y : E b
          e e' : Trivialization F proj
          H : Disjoint e.baseSet e'.baseSet
          x : Prod B F
          hx : Membership.mem (Min.min (SProd.sprod e.baseSet Set.univ) (SProd.sprod e'. …
          ⊢ Membership.mem Bot.bot x
        -/
        exact H.le_bot ⟨hx.1.1, hx.2.1⟩)
        /-
          🎉 no goals
        -/
  baseSet := e.baseSet ∪ e'.baseSet
  open_baseSet := IsOpen.union e.open_baseSet e'.open_baseSet
  source_eq := congr_arg₂ (· ∪ ·) e.source_eq e'.source_eq
  target_eq := (congr_arg₂ (· ∪ ·) e.target_eq e'.target_eq).trans union_prod.symm
  proj_toFun := by
    /-
      B : Type u_1
      F : Type u_2
      E : B → Type u_3
      Z : Type u_4
      inst✝³ : TopologicalSpace B
      inst✝² : TopologicalSpace F
      proj : Z → B
      inst✝¹ : TopologicalSpace Z
      inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
      e✝ : Trivialization F proj
      x : Z
      e'✝ : Trivialization F Bundle.TotalSpace.proj
      b : B
      y : E b
      e e' : Trivialization F proj
      H : Disjoint e.baseSet e'.baseSet
      ⊢ ∀ (p : Z), Membership.mem (e.disjointUnion e'.toPartialHomeomorph ⋯ ⋯).sourc …
    -/
    rintro p (hp | hp')
      /-
        case inl
        B : Type u_1
        F : Type u_2
        E : B → Type u_3
        Z : Type u_4
        inst✝³ : TopologicalSpace B
        inst✝² : TopologicalSpace F
        proj : Z → B
        inst✝¹ : TopologicalSpace Z
        inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
        e✝ : Trivialization F proj
        x : Z
        e'✝ : Trivialization F Bundle.TotalSpace.proj
        b : B
        y : E b
        e e' : Trivialization F proj
        H : Disjoint e.baseSet e'.baseSet
        p : Z
        hp : Membership.mem e.source p
        ⊢ Eq (↑(e.disjointUnion e'.toPartialHomeomorph ⋯ ⋯) p).1 (proj p)
      -/
    · show (e.source.piecewise e e' p).1 = proj p
      /-
        case inl
        B : Type u_1
        F : Type u_2
        E : B → Type u_3
        Z : Type u_4
        inst✝³ : TopologicalSpace B
        inst✝² : TopologicalSpace F
        proj : Z → B
        inst✝¹ : TopologicalSpace Z
        inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
        e✝ : Trivialization F proj
        x : Z
        e'✝ : Trivialization F Bundle.TotalSpace.proj
        b : B
        y : E b
        e e' : Trivialization F proj
        H : Disjoint e.baseSet e'.baseSet
        p : Z
        hp : Membership.mem e.source p
        ⊢ Eq (e.source.piecewise (↑e) (↑e') p).1 (proj p)
      -/
                                              /-
                                                🎉 no goals
                                              -/
      rw [piecewise_eq_of_mem, e.coe_fst] <;> exact hp
                                              /-
                                                🎉 no goals
                                              -/
      /-
        case inr
        B : Type u_1
        F : Type u_2
        E : B → Type u_3
        Z : Type u_4
        inst✝³ : TopologicalSpace B
        inst✝² : TopologicalSpace F
        proj : Z → B
        inst✝¹ : TopologicalSpace Z
        inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
        e✝ : Trivialization F proj
        x : Z
        e'✝ : Trivialization F Bundle.TotalSpace.proj
        b : B
        y : E b
        e e' : Trivialization F proj
        H : Disjoint e.baseSet e'.baseSet
        p : Z
        hp' : Membership.mem e'.source p
        ⊢ Eq (↑(e.disjointUnion e'.toPartialHomeomorph ⋯ ⋯) p).1 (proj p)
      -/
    · show (e.source.piecewise e e' p).1 = proj p
      /-
        case inr
        B : Type u_1
        F : Type u_2
        E : B → Type u_3
        Z : Type u_4
        inst✝³ : TopologicalSpace B
        inst✝² : TopologicalSpace F
        proj : Z → B
        inst✝¹ : TopologicalSpace Z
        inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
        e✝ : Trivialization F proj
        x : Z
        e'✝ : Trivialization F Bundle.TotalSpace.proj
        b : B
        y : E b
        e e' : Trivialization F proj
        H : Disjoint e.baseSet e'.baseSet
        p : Z
        hp' : Membership.mem e'.source p
        ⊢ Eq (e.source.piecewise (↑e) (↑e') p).1 (proj p)
      -/
      rw [piecewise_eq_of_not_mem, e'.coe_fst hp']
      /-
        case inr.hi
        B : Type u_1
        F : Type u_2
        E : B → Type u_3
        Z : Type u_4
        inst✝³ : TopologicalSpace B
        inst✝² : TopologicalSpace F
        proj : Z → B
        inst✝¹ : TopologicalSpace Z
        inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
        e✝ : Trivialization F proj
        x : Z
        e'✝ : Trivialization F Bundle.TotalSpace.proj
        b : B
        y : E b
        e e' : Trivialization F proj
        H : Disjoint e.baseSet e'.baseSet
        p : Z
        hp' : Membership.mem e'.source p
        ⊢ Not (Membership.mem e.source p)
      -/
      simp only [source_eq] at hp' ⊢
      /-
        case inr.hi
        B : Type u_1
        F : Type u_2
        E : B → Type u_3
        Z : Type u_4
        inst✝³ : TopologicalSpace B
        inst✝² : TopologicalSpace F
        proj : Z → B
        inst✝¹ : TopologicalSpace Z
        inst✝ : TopologicalSpace (Bundle.TotalSpace F E)
        e✝ : Trivialization F proj
        x : Z
        e'✝ : Trivialization F Bundle.TotalSpace.proj
        b : B
        y : E b
        e e' : Trivialization F proj
        H : Disjoint e.baseSet e'.baseSet
        p : Z
        hp' : Membership.mem (Set.preimage proj e'.baseSet) p
        ⊢ Not (Membership.mem (Set.preimage proj e.baseSet) p)
      -/
      exact fun h => H.le_bot ⟨h, hp'⟩
      /-
        🎉 no goals
      -/



/-- Continuous multilinear maps over the ring `R`, from `∀ i, M₁ i` to `M₂` where `M₁ i` and `M₂`
are modules over `R` with a topological structure. In applications, there will be compatibility
conditions between the algebraic and the topological structures, but this is not needed for the
definition. -/
structure ContinuousMultilinearMap (R : Type u) {ι : Type v} (M₁ : ι → Type w₁) (M₂ : Type w₂)
  [Semiring R] [∀ i, AddCommMonoid (M₁ i)] [AddCommMonoid M₂] [∀ i, Module R (M₁ i)] [Module R M₂]
  [∀ i, TopologicalSpace (M₁ i)] [TopologicalSpace M₂] extends MultilinearMap R M₁ M₂ where
  cont : Continuous toFun


@[inherit_doc]
notation:25 M "[×" n "]→L[" R "] " M' => ContinuousMultilinearMap R (fun i : Fin n => M) M'


theorem toMultilinearMap_injective :
    Function.Injective
      (ContinuousMultilinearMap.toMultilinearMap :
        ContinuousMultilinearMap R M₁ M₂ → MultilinearMap R M₁ M₂)
                              /-
                                R : Type u
                                ι : Type v
                                M₁ : ι → Type w₁
                                M₂ : Type w₂
                                inst✝⁶ : Semiring R
                                inst✝⁵ : (i : ι) → AddCommMonoid (M₁ i)
                                inst✝⁴ : AddCommMonoid M₂
                                inst✝³ : (i : ι) → Module R (M₁ i)
                                inst✝² : Module R M₂
                                inst✝¹ : (i : ι) → TopologicalSpace (M₁ i)
                                inst✝ : TopologicalSpace M₂
                                f : MultilinearMap R M₁ M₂
                                hf : Continuous f.toFun
                                g : MultilinearMap R M₁ M₂
                                hg : Continuous g.toFun
                                h : Eq { toMultilinearMap := f, cont := hf }.toMultilinearMap { toMultilinearM …
                                ⊢ Eq { toMultilinearMap := f, cont := hf } { toMultilinearMap := g, cont := hg }
                              -/
  | ⟨f, hf⟩, ⟨g, hg⟩, h => by subst h; rfl
                                       /-
                                         🎉 no goals
                                       -/


instance funLike : FunLike (ContinuousMultilinearMap R M₁ M₂) (∀ i, M₁ i) M₂ where
  coe f := f.toFun
  coe_injective' _ _ h := toMultilinearMap_injective <| MultilinearMap.coe_injective h


instance continuousMapClass :
    ContinuousMapClass (ContinuousMultilinearMap R M₁ M₂) (∀ i, M₁ i) M₂ where
  map_continuous := ContinuousMultilinearMap.cont


/-- See Note [custom simps projection]. We need to specify this projection explicitly in this case,
  because it is a composition of multiple projections. -/
def Simps.apply (L₁ : ContinuousMultilinearMap R M₁ M₂) (v : ∀ i, M₁ i) : M₂ :=
  L₁ v


@[continuity]
theorem coe_continuous : Continuous (f : (∀ i, M₁ i) → M₂) :=
  f.cont


@[simp]
theorem coe_coe : (f.toMultilinearMap : (∀ i, M₁ i) → M₂) = f :=
  rfl


@[ext]
theorem ext {f f' : ContinuousMultilinearMap R M₁ M₂} (H : ∀ x, f x = f' x) : f = f' :=
  DFunLike.ext _ _ H


@[simp]
theorem map_update_add [DecidableEq ι] (m : ∀ i, M₁ i) (i : ι) (x y : M₁ i) :
    f (update m i (x + y)) = f (update m i x) + f (update m i y) :=
  f.map_update_add' m i x y


@[deprecated (since := "2024-11-03")]
protected alias map_add := ContinuousMultilinearMap.map_update_add


@[simp]
theorem map_update_smul [DecidableEq ι] (m : ∀ i, M₁ i) (i : ι) (c : R) (x : M₁ i) :
    f (update m i (c • x)) = c • f (update m i x) :=
  f.map_update_smul' m i c x


@[deprecated (since := "2024-11-03")]
protected alias map_smul := ContinuousMultilinearMap.map_update_smul


theorem map_coord_zero {m : ∀ i, M₁ i} (i : ι) (h : m i = 0) : f m = 0 :=
  f.toMultilinearMap.map_coord_zero i h


@[simp]
theorem map_zero [Nonempty ι] : f 0 = 0 :=
  f.toMultilinearMap.map_zero


instance : Zero (ContinuousMultilinearMap R M₁ M₂) :=
  ⟨{ (0 : MultilinearMap R M₁ M₂) with cont := continuous_const }⟩


instance : Inhabited (ContinuousMultilinearMap R M₁ M₂) :=
  ⟨0⟩


@[simp]
theorem zero_apply (m : ∀ i, M₁ i) : (0 : ContinuousMultilinearMap R M₁ M₂) m = 0 :=
  rfl


@[simp]
theorem toMultilinearMap_zero : (0 : ContinuousMultilinearMap R M₁ M₂).toMultilinearMap = 0 :=
  rfl


instance : SMul R' (ContinuousMultilinearMap A M₁ M₂) :=
  ⟨fun c f => { c • f.toMultilinearMap with cont := f.cont.const_smul c }⟩


@[simp]
theorem smul_apply (f : ContinuousMultilinearMap A M₁ M₂) (c : R') (m : ∀ i, M₁ i) :
    (c • f) m = c • f m :=
  rfl


@[simp]
theorem toMultilinearMap_smul (c : R') (f : ContinuousMultilinearMap A M₁ M₂) :
    (c • f).toMultilinearMap = c • f.toMultilinearMap :=
  rfl


instance [SMulCommClass R' R'' M₂] : SMulCommClass R' R'' (ContinuousMultilinearMap A M₁ M₂) :=
  ⟨fun _ _ _ => ext fun _ => smul_comm _ _ _⟩


instance [SMul R' R''] [IsScalarTower R' R'' M₂] :
    IsScalarTower R' R'' (ContinuousMultilinearMap A M₁ M₂) :=
  ⟨fun _ _ _ => ext fun _ => smul_assoc _ _ _⟩


instance [DistribMulAction R'ᵐᵒᵖ M₂] [IsCentralScalar R' M₂] :
    IsCentralScalar R' (ContinuousMultilinearMap A M₁ M₂) :=
  ⟨fun _ _ => ext fun _ => op_smul_eq_smul _ _⟩


instance : MulAction R' (ContinuousMultilinearMap A M₁ M₂) :=
  Function.Injective.mulAction toMultilinearMap toMultilinearMap_injective fun _ _ => rfl


instance : Add (ContinuousMultilinearMap R M₁ M₂) :=
  ⟨fun f f' => ⟨f.toMultilinearMap + f'.toMultilinearMap, f.cont.add f'.cont⟩⟩


@[simp]
theorem add_apply (m : ∀ i, M₁ i) : (f + f') m = f m + f' m :=
  rfl


@[simp]
theorem toMultilinearMap_add (f g : ContinuousMultilinearMap R M₁ M₂) :
    (f + g).toMultilinearMap = f.toMultilinearMap + g.toMultilinearMap :=
  rfl


instance addCommMonoid : AddCommMonoid (ContinuousMultilinearMap R M₁ M₂) :=
  toMultilinearMap_injective.addCommMonoid _ rfl (fun _ _ => rfl) fun _ _ => rfl


/-- Evaluation of a `ContinuousMultilinearMap` at a vector as an `AddMonoidHom`. -/
def applyAddHom (m : ∀ i, M₁ i) : ContinuousMultilinearMap R M₁ M₂ →+ M₂ where
  toFun f := f m
  map_zero' := rfl
  map_add' _ _ := rfl


@[simp]
theorem sum_apply {α : Type*} (f : α → ContinuousMultilinearMap R M₁ M₂) (m : ∀ i, M₁ i)
    {s : Finset α} : (∑ a ∈ s, f a) m = ∑ a ∈ s, f a m :=
  map_sum (applyAddHom m) f s


/-- If `f` is a continuous multilinear map, then `f.toContinuousLinearMap m i` is the continuous
linear map obtained by fixing all coordinates but `i` equal to those of `m`, and varying the
`i`-th coordinate. -/
@[simps!] def toContinuousLinearMap [DecidableEq ι] (m : ∀ i, M₁ i) (i : ι) : M₁ i →L[R] M₂ :=
  { f.toMultilinearMap.toLinearMap m i with
    cont := f.cont.comp (continuous_const.update i continuous_id) }


/-- The cartesian product of two continuous multilinear maps, as a continuous multilinear map. -/
def prod (f : ContinuousMultilinearMap R M₁ M₂) (g : ContinuousMultilinearMap R M₁ M₃) :
    ContinuousMultilinearMap R M₁ (M₂ × M₃) :=
  { f.toMultilinearMap.prod g.toMultilinearMap with cont := f.cont.prod_mk g.cont }


@[simp]
theorem prod_apply (f : ContinuousMultilinearMap R M₁ M₂) (g : ContinuousMultilinearMap R M₁ M₃)
    (m : ∀ i, M₁ i) : (f.prod g) m = (f m, g m) :=
  rfl


/-- Combine a family of continuous multilinear maps with the same domain and codomains `M' i` into a
continuous multilinear map taking values in the space of functions `∀ i, M' i`. -/
def pi {ι' : Type*} {M' : ι' → Type*} [∀ i, AddCommMonoid (M' i)] [∀ i, TopologicalSpace (M' i)]
    [∀ i, Module R (M' i)] (f : ∀ i, ContinuousMultilinearMap R M₁ (M' i)) :
    ContinuousMultilinearMap R M₁ (∀ i, M' i) where
  cont := continuous_pi fun i => (f i).coe_continuous
  toMultilinearMap := MultilinearMap.pi fun i => (f i).toMultilinearMap


@[simp]
theorem coe_pi {ι' : Type*} {M' : ι' → Type*} [∀ i, AddCommMonoid (M' i)]
    [∀ i, TopologicalSpace (M' i)] [∀ i, Module R (M' i)]
    (f : ∀ i, ContinuousMultilinearMap R M₁ (M' i)) : ⇑(pi f) = fun m j => f j m :=
  rfl


theorem pi_apply {ι' : Type*} {M' : ι' → Type*} [∀ i, AddCommMonoid (M' i)]
    [∀ i, TopologicalSpace (M' i)] [∀ i, Module R (M' i)]
    (f : ∀ i, ContinuousMultilinearMap R M₁ (M' i)) (m : ∀ i, M₁ i) (j : ι') : pi f m j = f j m :=
  rfl


/-- Restrict the codomain of a continuous multilinear map to a submodule. -/
@[simps! toMultilinearMap apply_coe]
def codRestrict (f : ContinuousMultilinearMap R M₁ M₂) (p : Submodule R M₂) (h : ∀ v, f v ∈ p) :
    ContinuousMultilinearMap R M₁ p :=
  ⟨f.1.codRestrict p h, f.cont.subtype_mk _⟩


/-- The natural equivalence between continuous linear maps from `M₂` to `M₃`
and continuous 1-multilinear maps from `M₂` to `M₃`. -/
@[simps! apply_toMultilinearMap apply_apply symm_apply_apply]
def ofSubsingleton [Subsingleton ι] (i : ι) :
    (M₂ →L[R] M₃) ≃ ContinuousMultilinearMap R (fun _ : ι => M₂) M₃ where
  toFun f := ⟨MultilinearMap.ofSubsingleton R M₂ M₃ i f,
    (map_continuous f).comp (continuous_apply i)⟩
  invFun f := ⟨(MultilinearMap.ofSubsingleton R M₂ M₃ i).symm f.toMultilinearMap,
    (map_continuous f).comp <| continuous_pi fun _ ↦ continuous_id⟩
  left_inv _ := rfl
  right_inv f := toMultilinearMap_injective <|
    (MultilinearMap.ofSubsingleton R M₂ M₃ i).apply_symm_apply f.toMultilinearMap


/-- The constant map is multilinear when `ι` is empty. -/
@[simps! toMultilinearMap apply]
def constOfIsEmpty [IsEmpty ι] (m : M₂) : ContinuousMultilinearMap R M₁ M₂ where
  toMultilinearMap := MultilinearMap.constOfIsEmpty R _ m
  cont := continuous_const


/-- If `g` is continuous multilinear and `f` is a collection of continuous linear maps,
then `g (f₁ m₁, ..., fₙ mₙ)` is again a continuous multilinear map, that we call
`g.compContinuousLinearMap f`. -/
def compContinuousLinearMap (g : ContinuousMultilinearMap R M₁' M₄)
    (f : ∀ i : ι, M₁ i →L[R] M₁' i) : ContinuousMultilinearMap R M₁ M₄ :=
  { g.toMultilinearMap.compLinearMap fun i => (f i).toLinearMap with
    cont := g.cont.comp <| continuous_pi fun j => (f j).cont.comp <| continuous_apply _ }


@[simp]
theorem compContinuousLinearMap_apply (g : ContinuousMultilinearMap R M₁' M₄)
    (f : ∀ i : ι, M₁ i →L[R] M₁' i) (m : ∀ i, M₁ i) :
    g.compContinuousLinearMap f m = g fun i => f i <| m i :=
  rfl


/-- Composing a continuous multilinear map with a continuous linear map gives again a
continuous multilinear map. -/
def _root_.ContinuousLinearMap.compContinuousMultilinearMap (g : M₂ →L[R] M₃)
    (f : ContinuousMultilinearMap R M₁ M₂) : ContinuousMultilinearMap R M₁ M₃ :=
  { g.toLinearMap.compMultilinearMap f.toMultilinearMap with cont := g.cont.comp f.cont }


@[simp]
theorem _root_.ContinuousLinearMap.compContinuousMultilinearMap_coe (g : M₂ →L[R] M₃)
    (f : ContinuousMultilinearMap R M₁ M₂) :
    (g.compContinuousMultilinearMap f : (∀ i, M₁ i) → M₃) =
      (g : M₂ → M₃) ∘ (f : (∀ i, M₁ i) → M₂) := by
  /-
    R : Type u
    ι : Type v
    M₁ : ι → Type w₁
    M₂ : Type w₂
    M₃ : Type w₃
    inst✝⁹ : Semiring R
    inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁷ : AddCommMonoid M₂
    inst✝⁶ : AddCommMonoid M₃
    inst✝⁵ : (i : ι) → Module R (M₁ i)
    inst✝⁴ : Module R M₂
    inst✝³ : Module R M₃
    inst✝² : (i : ι) → TopologicalSpace (M₁ i)
    inst✝¹ : TopologicalSpace M₂
    inst✝ : TopologicalSpace M₃
    g : ContinuousLinearMap (RingHom.id R) M₂ M₃
    f : ContinuousMultilinearMap R M₁ M₂
    ⊢ Eq (⇑(g.compContinuousMultilinearMap f)) (Function.comp ⇑g ⇑f)
  -/
  ext m
  /-
    case h
    R : Type u
    ι : Type v
    M₁ : ι → Type w₁
    M₂ : Type w₂
    M₃ : Type w₃
    inst✝⁹ : Semiring R
    inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁷ : AddCommMonoid M₂
    inst✝⁶ : AddCommMonoid M₃
    inst✝⁵ : (i : ι) → Module R (M₁ i)
    inst✝⁴ : Module R M₂
    inst✝³ : Module R M₃
    inst✝² : (i : ι) → TopologicalSpace (M₁ i)
    inst✝¹ : TopologicalSpace M₂
    inst✝ : TopologicalSpace M₃
    g : ContinuousLinearMap (RingHom.id R) M₂ M₃
    f : ContinuousMultilinearMap R M₁ M₂
    m : (i : ι) → M₁ i
    ⊢ Eq ((g.compContinuousMultilinearMap f) m) (Function.comp (⇑g) (⇑f) m)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `ContinuousMultilinearMap.pi` as an `Equiv`. -/
@[simps]
def piEquiv {ι' : Type*} {M' : ι' → Type*} [∀ i, AddCommMonoid (M' i)]
    [∀ i, TopologicalSpace (M' i)] [∀ i, Module R (M' i)] :
    (∀ i, ContinuousMultilinearMap R M₁ (M' i)) ≃ ContinuousMultilinearMap R M₁ (∀ i, M' i) where
  toFun := ContinuousMultilinearMap.pi
  invFun f i := (ContinuousLinearMap.proj i : _ →L[R] M' i).compContinuousMultilinearMap f
  left_inv f := by
    /-
      R : Type u
      ι : Type v
      n : Nat
      M : Fin n.succ → Type w
      M₁ : ι → Type w₁
      M₁' : ι → Type w₁'
      M₂ : Type w₂
      M₃ : Type w₃
      M₄ : Type w₄
      inst✝²¹ : Semiring R
      inst✝²⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝¹⁹ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝¹⁸ : (i : ι) → AddCommMonoid (M₁' i)
      inst✝¹⁷ : AddCommMonoid M₂
      inst✝¹⁶ : AddCommMonoid M₃
      inst✝¹⁵ : AddCommMonoid M₄
      inst✝¹⁴ : (i : Fin n.succ) → Module R (M i)
      inst✝¹³ : (i : ι) → Module R (M₁ i)
      inst✝¹² : (i : ι) → Module R (M₁' i)
      inst✝¹¹ : Module R M₂
      inst✝¹⁰ : Module R M₃
      inst✝⁹ : Module R M₄
      inst✝⁸ : (i : Fin n.succ) → TopologicalSpace (M i)
      inst✝⁷ : (i : ι) → TopologicalSpace (M₁ i)
      inst✝⁶ : (i : ι) → TopologicalSpace (M₁' i)
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : TopologicalSpace M₃
      inst✝³ : TopologicalSpace M₄
      f✝ f' : ContinuousMultilinearMap R M₁ M₂
      ι' : Type u_1
      M' : ι' → Type u_2
      inst✝² : (i : ι') → AddCommMonoid (M' i)
      inst✝¹ : (i : ι') → TopologicalSpace (M' i)
      inst✝ : (i : ι') → Module R (M' i)
      f : (i : ι') → ContinuousMultilinearMap R M₁ (M' i)
      ⊢ Eq ((fun f i => (ContinuousLinearMap.proj i).compContinuousMultilinearMap f) …
    -/
    ext
    /-
      case h.H
      R : Type u
      ι : Type v
      n : Nat
      M : Fin n.succ → Type w
      M₁ : ι → Type w₁
      M₁' : ι → Type w₁'
      M₂ : Type w₂
      M₃ : Type w₃
      M₄ : Type w₄
      inst✝²¹ : Semiring R
      inst✝²⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝¹⁹ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝¹⁸ : (i : ι) → AddCommMonoid (M₁' i)
      inst✝¹⁷ : AddCommMonoid M₂
      inst✝¹⁶ : AddCommMonoid M₃
      inst✝¹⁵ : AddCommMonoid M₄
      inst✝¹⁴ : (i : Fin n.succ) → Module R (M i)
      inst✝¹³ : (i : ι) → Module R (M₁ i)
      inst✝¹² : (i : ι) → Module R (M₁' i)
      inst✝¹¹ : Module R M₂
      inst✝¹⁰ : Module R M₃
      inst✝⁹ : Module R M₄
      inst✝⁸ : (i : Fin n.succ) → TopologicalSpace (M i)
      inst✝⁷ : (i : ι) → TopologicalSpace (M₁ i)
      inst✝⁶ : (i : ι) → TopologicalSpace (M₁' i)
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : TopologicalSpace M₃
      inst✝³ : TopologicalSpace M₄
      f✝ f' : ContinuousMultilinearMap R M₁ M₂
      ι' : Type u_1
      M' : ι' → Type u_2
      inst✝² : (i : ι') → AddCommMonoid (M' i)
      inst✝¹ : (i : ι') → TopologicalSpace (M' i)
      inst✝ : (i : ι') → Module R (M' i)
      f : (i : ι') → ContinuousMultilinearMap R M₁ (M' i)
      x✝¹ : ι'
      x✝ : (i : ι) → M₁ i
      ⊢ Eq (((fun f i => (ContinuousLinearMap.proj i).compContinuousMultilinearMap f …
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      R : Type u
      ι : Type v
      n : Nat
      M : Fin n.succ → Type w
      M₁ : ι → Type w₁
      M₁' : ι → Type w₁'
      M₂ : Type w₂
      M₃ : Type w₃
      M₄ : Type w₄
      inst✝²¹ : Semiring R
      inst✝²⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝¹⁹ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝¹⁸ : (i : ι) → AddCommMonoid (M₁' i)
      inst✝¹⁷ : AddCommMonoid M₂
      inst✝¹⁶ : AddCommMonoid M₃
      inst✝¹⁵ : AddCommMonoid M₄
      inst✝¹⁴ : (i : Fin n.succ) → Module R (M i)
      inst✝¹³ : (i : ι) → Module R (M₁ i)
      inst✝¹² : (i : ι) → Module R (M₁' i)
      inst✝¹¹ : Module R M₂
      inst✝¹⁰ : Module R M₃
      inst✝⁹ : Module R M₄
      inst✝⁸ : (i : Fin n.succ) → TopologicalSpace (M i)
      inst✝⁷ : (i : ι) → TopologicalSpace (M₁ i)
      inst✝⁶ : (i : ι) → TopologicalSpace (M₁' i)
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : TopologicalSpace M₃
      inst✝³ : TopologicalSpace M₄
      f✝ f' : ContinuousMultilinearMap R M₁ M₂
      ι' : Type u_1
      M' : ι' → Type u_2
      inst✝² : (i : ι') → AddCommMonoid (M' i)
      inst✝¹ : (i : ι') → TopologicalSpace (M' i)
      inst✝ : (i : ι') → Module R (M' i)
      f : ContinuousMultilinearMap R M₁ ((i : ι') → M' i)
      ⊢ Eq (ContinuousMultilinearMap.pi ((fun f i => (ContinuousLinearMap.proj i).co …
    -/
    ext
    /-
      case H.h
      R : Type u
      ι : Type v
      n : Nat
      M : Fin n.succ → Type w
      M₁ : ι → Type w₁
      M₁' : ι → Type w₁'
      M₂ : Type w₂
      M₃ : Type w₃
      M₄ : Type w₄
      inst✝²¹ : Semiring R
      inst✝²⁰ : (i : Fin n.succ) → AddCommMonoid (M i)
      inst✝¹⁹ : (i : ι) → AddCommMonoid (M₁ i)
      inst✝¹⁸ : (i : ι) → AddCommMonoid (M₁' i)
      inst✝¹⁷ : AddCommMonoid M₂
      inst✝¹⁶ : AddCommMonoid M₃
      inst✝¹⁵ : AddCommMonoid M₄
      inst✝¹⁴ : (i : Fin n.succ) → Module R (M i)
      inst✝¹³ : (i : ι) → Module R (M₁ i)
      inst✝¹² : (i : ι) → Module R (M₁' i)
      inst✝¹¹ : Module R M₂
      inst✝¹⁰ : Module R M₃
      inst✝⁹ : Module R M₄
      inst✝⁸ : (i : Fin n.succ) → TopologicalSpace (M i)
      inst✝⁷ : (i : ι) → TopologicalSpace (M₁ i)
      inst✝⁶ : (i : ι) → TopologicalSpace (M₁' i)
      inst✝⁵ : TopologicalSpace M₂
      inst✝⁴ : TopologicalSpace M₃
      inst✝³ : TopologicalSpace M₄
      f✝ f' : ContinuousMultilinearMap R M₁ M₂
      ι' : Type u_1
      M' : ι' → Type u_2
      inst✝² : (i : ι') → AddCommMonoid (M' i)
      inst✝¹ : (i : ι') → TopologicalSpace (M' i)
      inst✝ : (i : ι') → Module R (M' i)
      f : ContinuousMultilinearMap R M₁ ((i : ι') → M' i)
      x✝¹ : (i : ι) → M₁ i
      x✝ : ι'
      ⊢ Eq ((ContinuousMultilinearMap.pi ((fun f i => (ContinuousLinearMap.proj i).c …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- An equivalence of the index set defines an equivalence between the spaces of continuous
multilinear maps. This is the forward map of this equivalence. -/
@[simps! toMultilinearMap apply]
nonrec def domDomCongr {ι' : Type*} (e : ι ≃ ι')
    (f : ContinuousMultilinearMap R (fun _ : ι => M₂) M₃) :
    ContinuousMultilinearMap R (fun _ : ι' => M₂) M₃ where
  toMultilinearMap := f.domDomCongr e
  cont := f.cont.comp <| continuous_pi fun _ => continuous_apply _


/-- An equivalence of the index set defines an equivalence between the spaces of continuous
multilinear maps. In case of normed spaces, this is a linear isometric equivalence, see
`ContinuousMultilinearMap.domDomCongrₗᵢ`. -/
@[simps]
def domDomCongrEquiv {ι' : Type*} (e : ι ≃ ι') :
    ContinuousMultilinearMap R (fun _ : ι => M₂) M₃ ≃
      ContinuousMultilinearMap R (fun _ : ι' => M₂) M₃ where
  toFun := domDomCongr e
  invFun := domDomCongr e.symm
                                /-
                                  R : Type u
                                  ι : Type v
                                  n : Nat
                                  M : Fin n.succ → Type w
                                  M₁ : ι → Type w₁
                                  M₁' : ι → Type w₁'
                                  M₂ : Type w₂
                                  M₃ : Type w₃
                                  M₄ : Type w₄
                                  inst✝¹⁸ : Semiring R
                                  inst✝¹⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
                                  inst✝¹⁶ : (i : ι) → AddCommMonoid (M₁ i)
                                  inst✝¹⁵ : (i : ι) → AddCommMonoid (M₁' i)
                                  inst✝¹⁴ : AddCommMonoid M₂
                                  inst✝¹³ : AddCommMonoid M₃
                                  inst✝¹² : AddCommMonoid M₄
                                  inst✝¹¹ : (i : Fin n.succ) → Module R (M i)
                                  inst✝¹⁰ : (i : ι) → Module R (M₁ i)
                                  inst✝⁹ : (i : ι) → Module R (M₁' i)
                                  inst✝⁸ : Module R M₂
                                  inst✝⁷ : Module R M₃
                                  inst✝⁶ : Module R M₄
                                  inst✝⁵ : (i : Fin n.succ) → TopologicalSpace (M i)
                                  inst✝⁴ : (i : ι) → TopologicalSpace (M₁ i)
                                  inst✝³ : (i : ι) → TopologicalSpace (M₁' i)
                                  inst✝² : TopologicalSpace M₂
                                  inst✝¹ : TopologicalSpace M₃
                                  inst✝ : TopologicalSpace M₄
                                  f f' : ContinuousMultilinearMap R M₁ M₂
                                  ι' : Type u_1
                                  e : Equiv ι ι'
                                  x✝¹ : ContinuousMultilinearMap R (fun x => M₂) M₃
                                  x✝ : ι → M₂
                                  ⊢ Eq ((ContinuousMultilinearMap.domDomCongr e.symm (ContinuousMultilinearMap.d …
                                -/
  left_inv _ := ext fun _ => by simp
                                /-
                                  🎉 no goals
                                -/
                                 /-
                                   R : Type u
                                   ι : Type v
                                   n : Nat
                                   M : Fin n.succ → Type w
                                   M₁ : ι → Type w₁
                                   M₁' : ι → Type w₁'
                                   M₂ : Type w₂
                                   M₃ : Type w₃
                                   M₄ : Type w₄
                                   inst✝¹⁸ : Semiring R
                                   inst✝¹⁷ : (i : Fin n.succ) → AddCommMonoid (M i)
                                   inst✝¹⁶ : (i : ι) → AddCommMonoid (M₁ i)
                                   inst✝¹⁵ : (i : ι) → AddCommMonoid (M₁' i)
                                   inst✝¹⁴ : AddCommMonoid M₂
                                   inst✝¹³ : AddCommMonoid M₃
                                   inst✝¹² : AddCommMonoid M₄
                                   inst✝¹¹ : (i : Fin n.succ) → Module R (M i)
                                   inst✝¹⁰ : (i : ι) → Module R (M₁ i)
                                   inst✝⁹ : (i : ι) → Module R (M₁' i)
                                   inst✝⁸ : Module R M₂
                                   inst✝⁷ : Module R M₃
                                   inst✝⁶ : Module R M₄
                                   inst✝⁵ : (i : Fin n.succ) → TopologicalSpace (M i)
                                   inst✝⁴ : (i : ι) → TopologicalSpace (M₁ i)
                                   inst✝³ : (i : ι) → TopologicalSpace (M₁' i)
                                   inst✝² : TopologicalSpace M₂
                                   inst✝¹ : TopologicalSpace M₃
                                   inst✝ : TopologicalSpace M₄
                                   f f' : ContinuousMultilinearMap R M₁ M₂
                                   ι' : Type u_1
                                   e : Equiv ι ι'
                                   x✝¹ : ContinuousMultilinearMap R (fun x => M₂) M₃
                                   x✝ : ι' → M₂
                                   ⊢ Eq ((ContinuousMultilinearMap.domDomCongr e (ContinuousMultilinearMap.domDom …
                                 -/
  right_inv _ := ext fun _ => by simp
                                 /-
                                   🎉 no goals
                                 -/


/-- The derivative of a continuous multilinear map, as a continuous linear map
from `∀ i, M₁ i` to `M₂`; see `ContinuousMultilinearMap.hasFDerivAt`. -/
def linearDeriv : (∀ i, M₁ i) →L[R] M₂ := ∑ i : ι, (f.toContinuousLinearMap x i).comp (.proj i)


@[simp]
lemma linearDeriv_apply : f.linearDeriv x y = ∑ i, f (Function.update x i (y i)) := by
  /-
    R : Type u
    ι : Type v
    M₁ : ι → Type w₁
    M₂ : Type w₂
    inst✝⁹ : Semiring R
    inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁷ : AddCommMonoid M₂
    inst✝⁶ : (i : ι) → Module R (M₁ i)
    inst✝⁵ : Module R M₂
    inst✝⁴ : (i : ι) → TopologicalSpace (M₁ i)
    inst✝³ : TopologicalSpace M₂
    f : ContinuousMultilinearMap R M₁ M₂
    inst✝² : ContinuousAdd M₂
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    x y : (i : ι) → M₁ i
    ⊢ Eq ((f.linearDeriv x) y) (Finset.univ.sum fun i => f (Function.update x i (y …
  -/
  unfold linearDeriv toContinuousLinearMap
  simp only [ContinuousLinearMap.coe_sum', ContinuousLinearMap.coe_comp',
    ContinuousLinearMap.coe_mk', LinearMap.coe_mk, LinearMap.coe_toAddHom, Finset.sum_apply]
  /-
    R : Type u
    ι : Type v
    M₁ : ι → Type w₁
    M₂ : Type w₂
    inst✝⁹ : Semiring R
    inst✝⁸ : (i : ι) → AddCommMonoid (M₁ i)
    inst✝⁷ : AddCommMonoid M₂
    inst✝⁶ : (i : ι) → Module R (M₁ i)
    inst✝⁵ : Module R M₂
    inst✝⁴ : (i : ι) → TopologicalSpace (M₁ i)
    inst✝³ : TopologicalSpace M₂
    f : ContinuousMultilinearMap R M₁ M₂
    inst✝² : ContinuousAdd M₂
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    x y : (i : ι) → M₁ i
    ⊢ Eq (Finset.univ.sum fun c => Function.comp (⇑(f.toLinearMap x c)) (⇑(Continu …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- In the specific case of continuous multilinear maps on spaces indexed by `Fin (n+1)`, where one
can build an element of `(i : Fin (n+1)) → M i` using `cons`, one can express directly the
additivity of a multilinear map along the first variable. -/
theorem cons_add (f : ContinuousMultilinearMap R M M₂) (m : ∀ i : Fin n, M i.succ) (x y : M 0) :
    f (cons (x + y) m) = f (cons x m) + f (cons y m) :=
  f.toMultilinearMap.cons_add m x y


/-- In the specific case of continuous multilinear maps on spaces indexed by `Fin (n+1)`, where one
can build an element of `(i : Fin (n+1)) → M i` using `cons`, one can express directly the
multiplicativity of a multilinear map along the first variable. -/
theorem cons_smul (f : ContinuousMultilinearMap R M M₂) (m : ∀ i : Fin n, M i.succ) (c : R)
    (x : M 0) : f (cons (c • x) m) = c • f (cons x m) :=
  f.toMultilinearMap.cons_smul m c x


theorem map_piecewise_add [DecidableEq ι] (m m' : ∀ i, M₁ i) (t : Finset ι) :
    f (t.piecewise (m + m') m') = ∑ s ∈ t.powerset, f (s.piecewise m m') :=
  f.toMultilinearMap.map_piecewise_add _ _ _


/-- Additivity of a continuous multilinear map along all coordinates at the same time,
writing `f (m + m')` as the sum of `f (s.piecewise m m')` over all sets `s`. -/
theorem map_add_univ [DecidableEq ι] [Fintype ι] (m m' : ∀ i, M₁ i) :
    f (m + m') = ∑ s : Finset ι, f (s.piecewise m m') :=
  f.toMultilinearMap.map_add_univ _ _


/-- If `f` is continuous multilinear, then `f (Σ_{j₁ ∈ A₁} g₁ j₁, ..., Σ_{jₙ ∈ Aₙ} gₙ jₙ)` is the
sum of `f (g₁ (r 1), ..., gₙ (r n))` where `r` ranges over all functions with `r 1 ∈ A₁`, ...,
`r n ∈ Aₙ`. This follows from multilinearity by expanding successively with respect to each
coordinate. -/
theorem map_sum_finset [DecidableEq ι] :
    (f fun i => ∑ j ∈ A i, g i j) = ∑ r ∈ piFinset A, f fun i => g i (r i) :=
  f.toMultilinearMap.map_sum_finset _ _


/-- If `f` is continuous multilinear, then `f (Σ_{j₁} g₁ j₁, ..., Σ_{jₙ} gₙ jₙ)` is the sum of
`f (g₁ (r 1), ..., gₙ (r n))` where `r` ranges over all functions `r`. This follows from
multilinearity by expanding successively with respect to each coordinate. -/
theorem map_sum [DecidableEq ι] [∀ i, Fintype (α i)] :
    (f fun i => ∑ j, g i j) = ∑ r : ∀ i, α i, f fun i => g i (r i) :=
  f.toMultilinearMap.map_sum _


/-- Reinterpret an `A`-multilinear map as an `R`-multilinear map, if `A` is an algebra over `R`
and their actions on all involved modules agree with the action of `R` on `A`. -/
def restrictScalars (f : ContinuousMultilinearMap A M₁ M₂) : ContinuousMultilinearMap R M₁ M₂ where
  toMultilinearMap := f.toMultilinearMap.restrictScalars R
  cont := f.cont


@[simp]
theorem coe_restrictScalars (f : ContinuousMultilinearMap A M₁ M₂) : ⇑(f.restrictScalars R) = f :=
  rfl


@[simp]
theorem map_update_sub [DecidableEq ι] (m : ∀ i, M₁ i) (i : ι) (x y : M₁ i) :
    f (update m i (x - y)) = f (update m i x) - f (update m i y) :=
  f.toMultilinearMap.map_update_sub _ _ _ _


@[deprecated (since := "2024-11-03")]
protected alias map_sub := ContinuousMultilinearMap.map_update_sub


instance : Neg (ContinuousMultilinearMap R M₁ M₂) :=
  ⟨fun f => { -f.toMultilinearMap with cont := f.cont.neg }⟩


@[simp]
theorem neg_apply (m : ∀ i, M₁ i) : (-f) m = -f m :=
  rfl


instance : Sub (ContinuousMultilinearMap R M₁ M₂) :=
  ⟨fun f g => { f.toMultilinearMap - g.toMultilinearMap with cont := f.cont.sub g.cont }⟩


@[simp]
theorem sub_apply (m : ∀ i, M₁ i) : (f - f') m = f m - f' m :=
  rfl


instance : AddCommGroup (ContinuousMultilinearMap R M₁ M₂) :=
  toMultilinearMap_injective.addCommGroup _ rfl (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) fun _ _ => rfl


theorem map_piecewise_smul [DecidableEq ι] (c : ι → R) (m : ∀ i, M₁ i) (s : Finset ι) :
    f (s.piecewise (fun i => c i • m i) m) = (∏ i ∈ s, c i) • f m :=
  f.toMultilinearMap.map_piecewise_smul _ _ _


/-- Multiplicativity of a continuous multilinear map along all coordinates at the same time,
writing `f (fun i ↦ c i • m i)` as `(∏ i, c i) • f m`. -/
theorem map_smul_univ [Fintype ι] (c : ι → R) (m : ∀ i, M₁ i) :
    (f fun i => c i • m i) = (∏ i, c i) • f m :=
  f.toMultilinearMap.map_smul_univ _ _


instance [ContinuousAdd M₂] : DistribMulAction R' (ContinuousMultilinearMap A M₁ M₂) :=
  Function.Injective.distribMulAction
    { toFun := toMultilinearMap,
      map_zero' := toMultilinearMap_zero,
      map_add' := toMultilinearMap_add }
    toMultilinearMap_injective
    fun _ _ => rfl


/-- The space of continuous multilinear maps over an algebra over `R` is a module over `R`, for the
pointwise addition and scalar multiplication. -/
instance : Module R' (ContinuousMultilinearMap A M₁ M₂) :=
  Function.Injective.module _
    { toFun := toMultilinearMap,
      map_zero' := toMultilinearMap_zero,
      map_add' := toMultilinearMap_add }
    toMultilinearMap_injective fun _ _ => rfl


/-- Linear map version of the map `toMultilinearMap` associating to a continuous multilinear map
the corresponding multilinear map. -/
@[simps]
def toMultilinearMapLinear : ContinuousMultilinearMap A M₁ M₂ →ₗ[R'] MultilinearMap A M₁ M₂ where
  toFun := toMultilinearMap
  map_add' := toMultilinearMap_add
  map_smul' := toMultilinearMap_smul


/-- `ContinuousMultilinearMap.pi` as a `LinearEquiv`. -/
@[simps (config := { simpRhs := true })]
def piLinearEquiv {ι' : Type*} {M' : ι' → Type*} [∀ i, AddCommMonoid (M' i)]
    [∀ i, TopologicalSpace (M' i)] [∀ i, ContinuousAdd (M' i)] [∀ i, Module R' (M' i)]
    [∀ i, Module A (M' i)] [∀ i, SMulCommClass A R' (M' i)] [∀ i, ContinuousConstSMul R' (M' i)] :
    (∀ i, ContinuousMultilinearMap A M₁ (M' i)) ≃ₗ[R'] ContinuousMultilinearMap A M₁ (∀ i, M' i) :=
  { piEquiv with
    map_add' := fun _ _ => rfl
    map_smul' := fun _ _ => rfl }


/-- The continuous multilinear map on `A^ι`, where `A` is a normed commutative algebra
over `𝕜`, associating to `m` the product of all the `m i`.

See also `ContinuousMultilinearMap.mkPiAlgebraFin`. -/
protected def mkPiAlgebra : ContinuousMultilinearMap R (fun _ : ι => A) A where
  cont := continuous_finset_prod _ fun _ _ => continuous_apply _
  toMultilinearMap := MultilinearMap.mkPiAlgebra R ι A


@[simp]
theorem mkPiAlgebra_apply (m : ι → A) : ContinuousMultilinearMap.mkPiAlgebra R ι A m = ∏ i, m i :=
  rfl


/-- The continuous multilinear map on `A^n`, where `A` is a normed algebra over `𝕜`, associating to
`m` the product of all the `m i`.

See also: `ContinuousMultilinearMap.mkPiAlgebra`. -/
protected def mkPiAlgebraFin : A[×n]→L[R] A where
  cont := by
    /-
      R : Type u
      ι : Type v
      n : Nat
      M : Fin n.succ → Type w
      M₁ : ι → Type w₁
      M₁' : ι → Type w₁'
      M₂ : Type w₂
      M₃ : Type w₃
      M₄ : Type w₄
      A : Type u_1
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      inst✝¹ : TopologicalSpace A
      inst✝ : ContinuousMul A
      ⊢ Continuous (MultilinearMap.mkPiAlgebraFin R n A).toFun
    -/
    change Continuous fun m => (List.ofFn m).prod
    /-
      R : Type u
      ι : Type v
      n : Nat
      M : Fin n.succ → Type w
      M₁ : ι → Type w₁
      M₁' : ι → Type w₁'
      M₂ : Type w₂
      M₃ : Type w₃
      M₄ : Type w₄
      A : Type u_1
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      inst✝¹ : TopologicalSpace A
      inst✝ : ContinuousMul A
      ⊢ Continuous fun m => (List.ofFn m).prod
    -/
    simp_rw [List.ofFn_eq_map]
    /-
      R : Type u
      ι : Type v
      n : Nat
      M : Fin n.succ → Type w
      M₁ : ι → Type w₁
      M₁' : ι → Type w₁'
      M₂ : Type w₂
      M₃ : Type w₃
      M₄ : Type w₄
      A : Type u_1
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      inst✝¹ : TopologicalSpace A
      inst✝ : ContinuousMul A
      ⊢ Continuous fun m => (List.map m (List.finRange n)).prod
    -/
    exact continuous_list_prod _ fun i _ => continuous_apply _
    /-
      🎉 no goals
    -/
  toMultilinearMap := MultilinearMap.mkPiAlgebraFin R n A


@[simp]
theorem mkPiAlgebraFin_apply (m : Fin n → A) :
    ContinuousMultilinearMap.mkPiAlgebraFin R n A m = (List.ofFn m).prod :=
  rfl


/-- Given a continuous `R`-multilinear map `f` taking values in `R`, `f.smulRight z` is the
continuous multilinear map sending `m` to `f m • z`. -/
@[simps! toMultilinearMap apply]
def smulRight : ContinuousMultilinearMap R M₁ M₂ where
  toMultilinearMap := f.toMultilinearMap.smulRight z
  cont := f.cont.smul continuous_const


variable (R ι) in
/-- The canonical continuous multilinear map on `R^ι`, associating to `m` the product of all the
`m i` (multiplied by a fixed reference element `z` in the target module) -/
protected def mkPiRing (z : M) : ContinuousMultilinearMap R (fun _ : ι => R) M :=
  (ContinuousMultilinearMap.mkPiAlgebra R ι R).smulRight z



@[simp]
theorem mkPiRing_apply (z : M) (m : ι → R) :
    (ContinuousMultilinearMap.mkPiRing R ι z : (ι → R) → M) m = (∏ i, m i) • z :=
  rfl


theorem mkPiRing_apply_one_eq_self (f : ContinuousMultilinearMap R (fun _ : ι => R) M) :
    ContinuousMultilinearMap.mkPiRing R ι (f fun _ => 1) = f :=
  toMultilinearMap_injective f.toMultilinearMap.mkPiRing_apply_one_eq_self


theorem mkPiRing_eq_iff {z₁ z₂ : M} :
    ContinuousMultilinearMap.mkPiRing R ι z₁ = ContinuousMultilinearMap.mkPiRing R ι z₂ ↔
      z₁ = z₂ := by
  /-
    R : Type u
    ι : Type v
    M : Type u_1
    inst✝⁷ : Fintype ι
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : TopologicalSpace R
    inst✝² : TopologicalSpace M
    inst✝¹ : ContinuousMul R
    inst✝ : ContinuousSMul R M
    z₁ z₂ : M
    ⊢ Iff (Eq (ContinuousMultilinearMap.mkPiRing R ι z₁) (ContinuousMultilinearMap …
  -/
  rw [← toMultilinearMap_injective.eq_iff]
  /-
    R : Type u
    ι : Type v
    M : Type u_1
    inst✝⁷ : Fintype ι
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : TopologicalSpace R
    inst✝² : TopologicalSpace M
    inst✝¹ : ContinuousMul R
    inst✝ : ContinuousSMul R M
    z₁ z₂ : M
    ⊢ Iff (Eq (ContinuousMultilinearMap.mkPiRing R ι z₁).toMultilinearMap (Continu …
  -/
  exact MultilinearMap.mkPiRing_eq_iff
  /-
    🎉 no goals
  -/


theorem mkPiRing_zero : ContinuousMultilinearMap.mkPiRing R ι (0 : M) = 0 := by
  /-
    R : Type u
    ι : Type v
    M : Type u_1
    inst✝⁷ : Fintype ι
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : TopologicalSpace R
    inst✝² : TopologicalSpace M
    inst✝¹ : ContinuousMul R
    inst✝ : ContinuousSMul R M
    ⊢ Eq (ContinuousMultilinearMap.mkPiRing R ι 0) 0
  -/
  ext; rw [mkPiRing_apply, smul_zero, ContinuousMultilinearMap.zero_apply]
       /-
         🎉 no goals
       -/


theorem mkPiRing_eq_zero_iff (z : M) : ContinuousMultilinearMap.mkPiRing R ι z = 0 ↔ z = 0 := by
  /-
    R : Type u
    ι : Type v
    M : Type u_1
    inst✝⁷ : Fintype ι
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : TopologicalSpace R
    inst✝² : TopologicalSpace M
    inst✝¹ : ContinuousMul R
    inst✝ : ContinuousSMul R M
    z : M
    ⊢ Iff (Eq (ContinuousMultilinearMap.mkPiRing R ι z) 0) (Eq z 0)
  -/
  rw [← mkPiRing_zero, mkPiRing_eq_iff]
  /-
    🎉 no goals
  -/



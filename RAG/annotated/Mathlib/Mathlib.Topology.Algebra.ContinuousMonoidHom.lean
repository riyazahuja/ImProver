/-- The type of continuous additive monoid homomorphisms from `A` to `B`.

When possible, instead of parametrizing results over `(f : ContinuousAddMonoidHom A B)`,
you should parametrize
over `(F : Type*) [FunLike F A B] [ContinuousMapClass F A B] [AddMonoidHomClass F A B] (f : F)`.

When you extend this structure,
make sure to extend `ContinuousMapClass` and/or `AddMonoidHomClass`, if needed. -/
structure ContinuousAddMonoidHom (A B : Type*) [AddMonoid A] [AddMonoid B] [TopologicalSpace A]
  [TopologicalSpace B] extends A →+ B, C(A, B)


/-- The type of continuous monoid homomorphisms from `A` to `B`.

When possible, instead of parametrizing results over `(f : ContinuousMonoidHom A B)`,
you should parametrize
over `(F : Type*) [FunLike F A B] [ContinuousMapClass F A B] [MonoidHomClass F A B] (f : F)`.

When you extend this structure,
make sure to extend `ContinuousMapClass` and/or `MonoidHomClass`, if needed. -/
@[to_additive "The type of continuous additive monoid homomorphisms from `A` to `B`."]
structure ContinuousMonoidHom extends A →* B, C(A, B)


/-- `ContinuousAddMonoidHomClass F A B` states that `F` is a type of continuous additive monoid
homomorphisms.

Deprecated and changed from a `class` to a `structure`.
Use `[AddMonoidHomClass F A B] [ContinuousMapClass F A B]` instead. -/
structure ContinuousAddMonoidHomClass (A B : outParam Type*) [AddMonoid A] [AddMonoid B]
    [TopologicalSpace A] [TopologicalSpace B] [FunLike F A B]
    extends AddMonoidHomClass F A B, ContinuousMapClass F A B : Prop


/-- `ContinuousMonoidHomClass F A B` states that `F` is a type of continuous monoid
homomorphisms.

Deprecated and changed from a `class` to a `structure`.
Use `[MonoidHomClass F A B] [ContinuousMapClass F A B]` instead. -/
@[to_additive (attr := deprecated "Use `[MonoidHomClass F A B] [ContinuousMapClass F A B]` instead."
  (since := "2024-10-08"))]
structure ContinuousMonoidHomClass (A B : outParam Type*) [Monoid A] [Monoid B]
    [TopologicalSpace A] [TopologicalSpace B] [FunLike F A B]
    extends MonoidHomClass F A B, ContinuousMapClass F A B : Prop


@[to_additive]
instance instFunLike : FunLike (ContinuousMonoidHom A B) A B where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      F : Type u_1
      A : Type u_2
      B : Type u_3
      C : Type u_4
      D : Type u_5
      E : Type u_6
      inst✝¹⁰ : Monoid A
      inst✝⁹ : Monoid B
      inst✝⁸ : Monoid C
      inst✝⁷ : Monoid D
      inst✝⁶ : CommGroup E
      inst✝⁵ : TopologicalSpace A
      inst✝⁴ : TopologicalSpace B
      inst✝³ : TopologicalSpace C
      inst✝² : TopologicalSpace D
      inst✝¹ : TopologicalSpace E
      inst✝ : TopologicalGroup E
      f g : ContinuousMonoidHom A B
      h : Eq ((fun f => (↑f.toMonoidHom).toFun) f) ((fun f => (↑f.toMonoidHom).toFun …
      ⊢ Eq f g
    -/
    obtain ⟨⟨⟨ _ , _ ⟩, _⟩, _⟩ := f
    /-
      case mk.mk.mk
      F : Type u_1
      A : Type u_2
      B : Type u_3
      C : Type u_4
      D : Type u_5
      E : Type u_6
      inst✝¹⁰ : Monoid A
      inst✝⁹ : Monoid B
      inst✝⁸ : Monoid C
      inst✝⁷ : Monoid D
      inst✝⁶ : CommGroup E
      inst✝⁵ : TopologicalSpace A
      inst✝⁴ : TopologicalSpace B
      inst✝³ : TopologicalSpace C
      inst✝² : TopologicalSpace D
      inst✝¹ : TopologicalSpace E
      inst✝ : TopologicalGroup E
      g : ContinuousMonoidHom A B
      toFun✝ : A → B
      map_one'✝ : Eq (toFun✝ 1) 1
      map_mul'✝ : ∀ (x y : A), Eq ({ toFun := toFun✝, map_one' := map_one'✝ }.toFun  …
      continuous_toFun✝ : Continuous (↑{ toFun := toFun✝, map_one' := map_one'✝, map …
      h : Eq ((fun f => (↑f.toMonoidHom).toFun) { toFun := toFun✝, map_one' := map_o …
      ⊢ Eq { toFun := toFun✝, map_one' := map_one'✝, map_mul' := map_mul'✝, continuo …
    -/
    obtain ⟨⟨⟨ _ , _ ⟩, _⟩, _⟩ := g
    /-
      case mk.mk.mk.mk.mk.mk
      F : Type u_1
      A : Type u_2
      B : Type u_3
      C : Type u_4
      D : Type u_5
      E : Type u_6
      inst✝¹⁰ : Monoid A
      inst✝⁹ : Monoid B
      inst✝⁸ : Monoid C
      inst✝⁷ : Monoid D
      inst✝⁶ : CommGroup E
      inst✝⁵ : TopologicalSpace A
      inst✝⁴ : TopologicalSpace B
      inst✝³ : TopologicalSpace C
      inst✝² : TopologicalSpace D
      inst✝¹ : TopologicalSpace E
      inst✝ : TopologicalGroup E
      toFun✝¹ : A → B
      map_one'✝¹ : Eq (toFun✝¹ 1) 1
      map_mul'✝¹ : ∀ (x y : A), Eq ({ toFun := toFun✝¹, map_one' := map_one'✝¹ }.toF …
      continuous_toFun✝¹ : Continuous (↑{ toFun := toFun✝¹, map_one' := map_one'✝¹,  …
      toFun✝ : A → B
      map_one'✝ : Eq (toFun✝ 1) 1
      map_mul'✝ : ∀ (x y : A), Eq ({ toFun := toFun✝, map_one' := map_one'✝ }.toFun  …
      continuous_toFun✝ : Continuous (↑{ toFun := toFun✝, map_one' := map_one'✝, map …
      h : Eq ((fun f => (↑f.toMonoidHom).toFun) { toFun := toFun✝¹, map_one' := map_ …
      ⊢ Eq { toFun := toFun✝¹, map_one' := map_one'✝¹, map_mul' := map_mul'✝¹, conti …
    -/
    congr
    /-
      🎉 no goals
    -/


@[to_additive]
instance instMonoidHomClass : MonoidHomClass (ContinuousMonoidHom A B) A B where
  map_mul f := f.map_mul'
  map_one f := f.map_one'


@[to_additive]
instance instContinuousMapClass : ContinuousMapClass (ContinuousMonoidHom A B) A B where
  map_continuous f := f.continuous_toFun


@[to_additive (attr := ext)]
theorem ext {f g : ContinuousMonoidHom A B} (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext _ _ h


@[to_additive]
theorem toContinuousMap_injective : Injective (toContinuousMap : _ → C(A, B)) := fun f g h =>
            /-
              A : Type u_2
              B : Type u_3
              inst✝³ : Monoid A
              inst✝² : Monoid B
              inst✝¹ : TopologicalSpace A
              inst✝ : TopologicalSpace B
              f g : ContinuousMonoidHom A B
              h : Eq f.toContinuousMap g.toContinuousMap
              ⊢ ∀ (x : A), Eq (f x) (g x)
            -/
  ext <| by convert DFunLike.ext_iff.1 h
            /-
              🎉 no goals
            -/


@[deprecated (since := "2024-10-08")] protected alias mk' := mk


@[deprecated (since := "2024-10-08")]
protected alias _root_.ContinuousAddMonoidHom.mk' := ContinuousAddMonoidHom.mk


set_option linter.existingAttributeWarning false in
attribute [to_additive existing] ContinuousMonoidHom.mk'


/-- Composition of two continuous homomorphisms. -/
@[to_additive (attr := simps!) "Composition of two continuous homomorphisms."]
def comp (g : ContinuousMonoidHom B C) (f : ContinuousMonoidHom A B) : ContinuousMonoidHom A C :=
  ⟨g.toMonoidHom.comp f.toMonoidHom, (map_continuous g).comp (map_continuous f)⟩


/-- Product of two continuous homomorphisms on the same space. -/
@[to_additive (attr := simps!) prod "Product of two continuous homomorphisms on the same space."]
def prod (f : ContinuousMonoidHom A B) (g : ContinuousMonoidHom A C) :
    ContinuousMonoidHom A (B × C) :=
  ⟨f.toMonoidHom.prod g.toMonoidHom, f.continuous_toFun.prod_mk g.continuous_toFun⟩


/-- Product of two continuous homomorphisms on different spaces. -/
@[to_additive (attr := simps!) prodMap
  "Product of two continuous homomorphisms on different spaces."]
def prodMap (f : ContinuousMonoidHom A C) (g : ContinuousMonoidHom B D) :
    ContinuousMonoidHom (A × B) (C × D) :=
  ⟨f.toMonoidHom.prodMap g.toMonoidHom, f.continuous_toFun.prodMap g.continuous_toFun⟩


@[deprecated (since := "2024-10-05")] alias prod_map := prodMap

@[deprecated (since := "2024-10-05")]
alias _root_.ContinuousAddMonoidHom.sum_map := ContinuousAddMonoidHom.prodMap


set_option linter.existingAttributeWarning false in
attribute [to_additive existing] prod_map


/-- The trivial continuous homomorphism. -/
@[to_additive (attr := simps!) "The trivial continuous homomorphism."]
def one : ContinuousMonoidHom A B :=
  ⟨1, continuous_const⟩


@[to_additive]
instance : Inhabited (ContinuousMonoidHom A B) :=
  ⟨one A B⟩


/-- The identity continuous homomorphism. -/
@[to_additive (attr := simps!) "The identity continuous homomorphism."]
def id : ContinuousMonoidHom A A :=
  ⟨.id A, continuous_id⟩


/-- The continuous homomorphism given by projection onto the first factor. -/
@[to_additive (attr := simps!)
  "The continuous homomorphism given by projection onto the first factor."]
def fst : ContinuousMonoidHom (A × B) A :=
  ⟨MonoidHom.fst A B, continuous_fst⟩


/-- The continuous homomorphism given by projection onto the second factor. -/
@[to_additive (attr := simps!)
  "The continuous homomorphism given by projection onto the second factor."]
def snd : ContinuousMonoidHom (A × B) B :=
  ⟨MonoidHom.snd A B, continuous_snd⟩


/-- The continuous homomorphism given by inclusion of the first factor. -/
@[to_additive (attr := simps!)
  "The continuous homomorphism given by inclusion of the first factor."]
def inl : ContinuousMonoidHom A (A × B) :=
  prod (id A) (one A B)


/-- The continuous homomorphism given by inclusion of the second factor. -/
@[to_additive (attr := simps!)
  "The continuous homomorphism given by inclusion of the second factor."]
def inr : ContinuousMonoidHom B (A × B) :=
  prod (one B A) (id B)


/-- The continuous homomorphism given by the diagonal embedding. -/
@[to_additive (attr := simps!) "The continuous homomorphism given by the diagonal embedding."]
def diag : ContinuousMonoidHom A (A × A) :=
  prod (id A) (id A)


/-- The continuous homomorphism given by swapping components. -/
@[to_additive (attr := simps!) "The continuous homomorphism given by swapping components."]
def swap : ContinuousMonoidHom (A × B) (B × A) :=
  prod (snd A B) (fst A B)


/-- The continuous homomorphism given by multiplication. -/
@[to_additive (attr := simps!) "The continuous homomorphism given by addition."]
def mul : ContinuousMonoidHom (E × E) E :=
  ⟨mulMonoidHom, continuous_mul⟩


/-- The continuous homomorphism given by inversion. -/
@[to_additive (attr := simps!) "The continuous homomorphism given by negation."]
def inv : ContinuousMonoidHom E E :=
  ⟨invMonoidHom, continuous_inv⟩


/-- Coproduct of two continuous homomorphisms to the same space. -/
@[to_additive (attr := simps!) "Coproduct of two continuous homomorphisms to the same space."]
def coprod (f : ContinuousMonoidHom A E) (g : ContinuousMonoidHom B E) :
    ContinuousMonoidHom (A × B) E :=
  (mul E).comp (f.prodMap g)


@[to_additive]
instance : CommGroup (ContinuousMonoidHom A E) where
  mul f g := (mul E).comp (f.prod g)
  mul_comm f g := ext fun x => mul_comm (f x) (g x)
  mul_assoc f g h := ext fun x => mul_assoc (f x) (g x) (h x)
  one := one A E
  one_mul f := ext fun x => one_mul (f x)
  mul_one f := ext fun x => mul_one (f x)
  inv f := (inv E).comp f
  inv_mul_cancel f := ext fun x => inv_mul_cancel (f x)


@[to_additive]
instance : TopologicalSpace (ContinuousMonoidHom A B) :=
  TopologicalSpace.induced toContinuousMap ContinuousMap.compactOpen


@[to_additive]
theorem isInducing_toContinuousMap :
    IsInducing (toContinuousMap : ContinuousMonoidHom A B → C(A, B)) := ⟨rfl⟩


@[deprecated (since := "2024-10-28")] alias inducing_toContinuousMap := isInducing_toContinuousMap


@[to_additive]
theorem isEmbedding_toContinuousMap :
    IsEmbedding (toContinuousMap : ContinuousMonoidHom A B → C(A, B)) :=
  ⟨isInducing_toContinuousMap A B, toContinuousMap_injective⟩


@[deprecated (since := "2024-10-26")]
alias embedding_toContinuousMap := isEmbedding_toContinuousMap


@[to_additive]
instance instContinuousEvalConst : ContinuousEvalConst (ContinuousMonoidHom A B) A B :=
  /-
    F : Type u_1
    A : Type u_2
    B : Type u_3
    C : Type u_4
    D : Type u_5
    E : Type u_6
    inst✝¹⁰ : Monoid A
    inst✝⁹ : Monoid B
    inst✝⁸ : Monoid C
    inst✝⁷ : Monoid D
    inst✝⁶ : CommGroup E
    inst✝⁵ : TopologicalSpace A
    inst✝⁴ : TopologicalSpace B
    inst✝³ : TopologicalSpace C
    inst✝² : TopologicalSpace D
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalGroup E
    ⊢ ∀ (g : ContinuousMonoidHom A B), Eq ⇑g.toContinuousMap ⇑g
  -/
  .of_continuous_forget (isInducing_toContinuousMap A B).continuous
  /-
    🎉 no goals
  -/


@[to_additive]
instance instContinuousEval [LocallyCompactPair A B] :
    ContinuousEval (ContinuousMonoidHom A B) A B :=
  /-
    F : Type u_1
    A : Type u_2
    B : Type u_3
    C : Type u_4
    D : Type u_5
    E : Type u_6
    inst✝¹¹ : Monoid A
    inst✝¹⁰ : Monoid B
    inst✝⁹ : Monoid C
    inst✝⁸ : Monoid D
    inst✝⁷ : CommGroup E
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : TopologicalSpace B
    inst✝⁴ : TopologicalSpace C
    inst✝³ : TopologicalSpace D
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalGroup E
    inst✝ : LocallyCompactPair A B
    ⊢ ∀ (g : ContinuousMonoidHom A B), Eq ⇑g.toContinuousMap ⇑g
  -/
  .of_continuous_forget (isInducing_toContinuousMap A B).continuous
  /-
    🎉 no goals
  -/


@[to_additive]
lemma range_toContinuousMap :
    Set.range (toContinuousMap : ContinuousMonoidHom A B → C(A, B)) =
      {f : C(A, B) | f 1 = 1 ∧ ∀ x y, f (x * y) = f x * f y} := by
  /-
    A : Type u_2
    B : Type u_3
    inst✝³ : Monoid A
    inst✝² : Monoid B
    inst✝¹ : TopologicalSpace A
    inst✝ : TopologicalSpace B
    ⊢ Eq (Set.range ContinuousMonoidHom.toContinuousMap) (setOf fun f => And (Eq ( …
  -/
  refine Set.Subset.antisymm (Set.range_subset_iff.2 fun f ↦ ⟨map_one f, map_mul f⟩) ?_
  /-
    A : Type u_2
    B : Type u_3
    inst✝³ : Monoid A
    inst✝² : Monoid B
    inst✝¹ : TopologicalSpace A
    inst✝ : TopologicalSpace B
    ⊢ HasSubset.Subset (setOf fun f => And (Eq (f 1) 1) (∀ (x y : A), Eq (f (HMul. …
  -/
  rintro f ⟨h1, hmul⟩
  /-
    case intro
    A : Type u_2
    B : Type u_3
    inst✝³ : Monoid A
    inst✝² : Monoid B
    inst✝¹ : TopologicalSpace A
    inst✝ : TopologicalSpace B
    f : ContinuousMap A B
    h1 : Eq (f 1) 1
    hmul : ∀ (x y : A), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    ⊢ Membership.mem (Set.range ContinuousMonoidHom.toContinuousMap) f
  -/
  exact ⟨{ f with map_one' := h1, map_mul' := hmul }, rfl⟩
  /-
    🎉 no goals
  -/


@[to_additive]
theorem isClosedEmbedding_toContinuousMap [ContinuousMul B] [T2Space B] :
    IsClosedEmbedding (toContinuousMap : ContinuousMonoidHom A B → C(A, B)) where
  toIsEmbedding := isEmbedding_toContinuousMap A B
  isClosed_range := by
    /-
      A : Type u_2
      B : Type u_3
      inst✝⁵ : Monoid A
      inst✝⁴ : Monoid B
      inst✝³ : TopologicalSpace A
      inst✝² : TopologicalSpace B
      inst✝¹ : ContinuousMul B
      inst✝ : T2Space B
      ⊢ IsClosed (Set.range ContinuousMonoidHom.toContinuousMap)
    -/
    simp only [range_toContinuousMap, Set.setOf_and, Set.setOf_forall]
    refine .inter (isClosed_singleton.preimage (continuous_eval_const 1)) <|
      isClosed_iInter fun x ↦ isClosed_iInter fun y ↦ ?_
    exact isClosed_eq (continuous_eval_const (x * y)) <|
      .mul (continuous_eval_const x) (continuous_eval_const y)


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_toContinuousMap := isClosedEmbedding_toContinuousMap


@[to_additive]
instance [T2Space B] : T2Space (ContinuousMonoidHom A B) :=
  (isEmbedding_toContinuousMap A B).t2Space


@[to_additive]
instance : TopologicalGroup (ContinuousMonoidHom A E) :=
  let hi := isInducing_toContinuousMap A E
  let hc := hi.continuous
  { continuous_mul := hi.continuous_iff.mpr (continuous_mul.comp (Continuous.prodMap hc hc))
    continuous_inv := hi.continuous_iff.mpr (continuous_inv.comp hc) }


@[to_additive]
theorem continuous_of_continuous_uncurry {A : Type*} [TopologicalSpace A]
    (f : A → ContinuousMonoidHom B C) (h : Continuous (Function.uncurry fun x y => f x y)) :
    Continuous f :=
  (isInducing_toContinuousMap _ _).continuous_iff.mpr
    (ContinuousMap.continuous_of_continuous_uncurry _ h)


@[to_additive]
theorem continuous_comp [LocallyCompactSpace B] :
    Continuous fun f : ContinuousMonoidHom A B × ContinuousMonoidHom B C => f.2.comp f.1 :=
  (isInducing_toContinuousMap A C).continuous_iff.2 <|
    ContinuousMap.continuous_comp'.comp
      ((isInducing_toContinuousMap A B).prodMap (isInducing_toContinuousMap B C)).continuous


@[to_additive]
theorem continuous_comp_left (f : ContinuousMonoidHom A B) :
    Continuous fun g : ContinuousMonoidHom B C => g.comp f :=
  (isInducing_toContinuousMap A C).continuous_iff.2 <|
    f.toContinuousMap.continuous_precomp.comp (isInducing_toContinuousMap B C).continuous


@[to_additive]
theorem continuous_comp_right (f : ContinuousMonoidHom B C) :
    Continuous fun g : ContinuousMonoidHom A B => f.comp g :=
  (isInducing_toContinuousMap A C).continuous_iff.2 <|
    f.toContinuousMap.continuous_postcomp.comp (isInducing_toContinuousMap A B).continuous


/-- `ContinuousMonoidHom _ f` is a functor. -/
@[to_additive "`ContinuousAddMonoidHom _ f` is a functor."]
def compLeft (f : ContinuousMonoidHom A B) :
    ContinuousMonoidHom (ContinuousMonoidHom B E) (ContinuousMonoidHom A E) where
  toFun g := g.comp f
  map_one' := rfl
  map_mul' _g _h := rfl
  continuous_toFun := f.continuous_comp_left


/-- `ContinuousMonoidHom f _` is a functor. -/
@[to_additive "`ContinuousAddMonoidHom f _` is a functor."]
def compRight {B : Type*} [CommGroup B] [TopologicalSpace B] [TopologicalGroup B]
    (f : ContinuousMonoidHom B E) :
    ContinuousMonoidHom (ContinuousMonoidHom A B) (ContinuousMonoidHom A E) where
  toFun g := f.comp g
  map_one' := ext fun _a => map_one f
  map_mul' g h := ext fun a => map_mul f (g a) (h a)
  continuous_toFun := f.continuous_comp_right


@[to_additive]
theorem locallyCompactSpace_of_equicontinuousAt (U : Set X) (V : Set Y)
    (hU : IsCompact U) (hV : V ∈ nhds (1 : Y))
    (h : EquicontinuousAt (fun f : {f : X →* Y | Set.MapsTo f U V} ↦ (f : X → Y)) 1) :
    LocallyCompactSpace (ContinuousMonoidHom X Y) := by
  /-
    X : Type u_7
    Y : Type u_8
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : Group X
    inst✝⁵ : TopologicalGroup X
    inst✝⁴ : UniformSpace Y
    inst✝³ : CommGroup Y
    inst✝² : UniformGroup Y
    inst✝¹ : T0Space Y
    inst✝ : CompactSpace Y
    U : Set X
    V : Set Y
    hU : IsCompact U
    hV : Membership.mem (nhds 1) V
    h : EquicontinuousAt (fun f => ⇑↑f) 1
    ⊢ LocallyCompactSpace (ContinuousMonoidHom X Y)
  -/
  replace h := equicontinuous_of_equicontinuousAt_one _ h
  /-
    X : Type u_7
    Y : Type u_8
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : Group X
    inst✝⁵ : TopologicalGroup X
    inst✝⁴ : UniformSpace Y
    inst✝³ : CommGroup Y
    inst✝² : UniformGroup Y
    inst✝¹ : T0Space Y
    inst✝ : CompactSpace Y
    U : Set X
    V : Set Y
    hU : IsCompact U
    hV : Membership.mem (nhds 1) V
    h : Equicontinuous (Function.comp DFunLike.coe Subtype.val)
    ⊢ LocallyCompactSpace (ContinuousMonoidHom X Y)
  -/
  obtain ⟨W, hWo, hWV, hWc⟩ := local_compact_nhds hV
  /-
    case intro.intro.intro
    X : Type u_7
    Y : Type u_8
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : Group X
    inst✝⁵ : TopologicalGroup X
    inst✝⁴ : UniformSpace Y
    inst✝³ : CommGroup Y
    inst✝² : UniformGroup Y
    inst✝¹ : T0Space Y
    inst✝ : CompactSpace Y
    U : Set X
    V : Set Y
    hU : IsCompact U
    hV : Membership.mem (nhds 1) V
    h : Equicontinuous (Function.comp DFunLike.coe Subtype.val)
    W : Set Y
    hWo : Membership.mem (nhds 1) W
    hWV : HasSubset.Subset W V
    hWc : IsCompact W
    ⊢ LocallyCompactSpace (ContinuousMonoidHom X Y)
  -/
  let S1 : Set (X →* Y) := {f | Set.MapsTo f U W}
  /-
    case intro.intro.intro
    X : Type u_7
    Y : Type u_8
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : Group X
    inst✝⁵ : TopologicalGroup X
    inst✝⁴ : UniformSpace Y
    inst✝³ : CommGroup Y
    inst✝² : UniformGroup Y
    inst✝¹ : T0Space Y
    inst✝ : CompactSpace Y
    U : Set X
    V : Set Y
    hU : IsCompact U
    hV : Membership.mem (nhds 1) V
    h : Equicontinuous (Function.comp DFunLike.coe Subtype.val)
    W : Set Y
    hWo : Membership.mem (nhds 1) W
    hWV : HasSubset.Subset W V
    hWc : IsCompact W
    S1 : Set (MonoidHom X Y) := setOf fun f => Set.MapsTo (⇑f) U W
    ⊢ LocallyCompactSpace (ContinuousMonoidHom X Y)
  -/
  let S2 : Set (ContinuousMonoidHom X Y) := {f | Set.MapsTo f U W}
  /-
    case intro.intro.intro
    X : Type u_7
    Y : Type u_8
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : Group X
    inst✝⁵ : TopologicalGroup X
    inst✝⁴ : UniformSpace Y
    inst✝³ : CommGroup Y
    inst✝² : UniformGroup Y
    inst✝¹ : T0Space Y
    inst✝ : CompactSpace Y
    U : Set X
    V : Set Y
    hU : IsCompact U
    hV : Membership.mem (nhds 1) V
    h : Equicontinuous (Function.comp DFunLike.coe Subtype.val)
    W : Set Y
    hWo : Membership.mem (nhds 1) W
    hWV : HasSubset.Subset W V
    hWc : IsCompact W
    S1 : Set (MonoidHom X Y) := setOf fun f => Set.MapsTo (⇑f) U W
    S2 : Set (ContinuousMonoidHom X Y) := setOf fun f => Set.MapsTo (⇑f) U W
    ⊢ LocallyCompactSpace (ContinuousMonoidHom X Y)
  -/
  let S3 : Set C(X, Y) := (↑) '' S2
  /-
    case intro.intro.intro
    X : Type u_7
    Y : Type u_8
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : Group X
    inst✝⁵ : TopologicalGroup X
    inst✝⁴ : UniformSpace Y
    inst✝³ : CommGroup Y
    inst✝² : UniformGroup Y
    inst✝¹ : T0Space Y
    inst✝ : CompactSpace Y
    U : Set X
    V : Set Y
    hU : IsCompact U
    hV : Membership.mem (nhds 1) V
    h : Equicontinuous (Function.comp DFunLike.coe Subtype.val)
    W : Set Y
    hWo : Membership.mem (nhds 1) W
    hWV : HasSubset.Subset W V
    hWc : IsCompact W
    S1 : Set (MonoidHom X Y) := setOf fun f => Set.MapsTo (⇑f) U W
    S2 : Set (ContinuousMonoidHom X Y) := setOf fun f => Set.MapsTo (⇑f) U W
    S3 : Set (ContinuousMap X Y) := Set.image _root_.toContinuousMap S2
    ⊢ LocallyCompactSpace (ContinuousMonoidHom X Y)
  -/
  let S4 : Set (X → Y) := (↑) '' S3
  replace h : Equicontinuous ((↑) : S1 → X → Y) :=
    h.comp (Subtype.map _root_.id fun f hf ↦ hf.mono_right hWV)
  have hS4 : S4 = (↑) '' S1 := by
    ext
    constructor
    · rintro ⟨-, ⟨f, hf, rfl⟩, rfl⟩
      exact ⟨f, hf, rfl⟩
    · rintro ⟨f, hf, rfl⟩
      exact ⟨⟨f, h.continuous ⟨f, hf⟩⟩, ⟨⟨f, h.continuous ⟨f, hf⟩⟩, hf, rfl⟩, rfl⟩
  replace h : Equicontinuous ((↑) : S3 → X → Y) := by
    rw [equicontinuous_iff_range, ← Set.image_eq_range] at h ⊢
    rwa [← hS4] at h
  replace hS4 : S4 = Set.pi U (fun _ ↦ W) ∩ Set.range ((↑) : (X →* Y) → (X → Y)) := by
    simp_rw [hS4, Set.ext_iff, Set.mem_image, S1, Set.mem_setOf_eq]
    exact fun f ↦ ⟨fun ⟨g, hg, hf⟩ ↦ hf ▸ ⟨hg, g, rfl⟩, fun ⟨hg, g, hf⟩ ↦ ⟨g, hf ▸ hg, hf⟩⟩
  replace hS4 : IsClosed S4 :=
    hS4.symm ▸ (isClosed_set_pi (fun _ _ ↦ hWc.isClosed)).inter (MonoidHom.isClosed_range_coe X Y)
  have hS2 : (interior S2).Nonempty := by
    let T : Set (ContinuousMonoidHom X Y) := {f | Set.MapsTo f U (interior W)}
    have h1 : T.Nonempty := ⟨1, fun _ _ ↦ mem_interior_iff_mem_nhds.mpr hWo⟩
    have h2 : T ⊆ S2 := fun f hf ↦ hf.mono_right interior_subset
    have h3 : IsOpen T := isOpen_induced (ContinuousMap.isOpen_setOf_mapsTo hU isOpen_interior)
    exact h1.mono (interior_maximal h2 h3)
  exact TopologicalSpace.PositiveCompacts.locallyCompactSpace_of_group
    ⟨⟨S2, (isInducing_toContinuousMap X Y).isCompact_iff.mpr
      (ArzelaAscoli.isCompact_of_equicontinuous S3 hS4.isCompact h)⟩, hS2⟩


@[to_additive]
theorem locallyCompactSpace_of_hasBasis (V : ℕ → Set Y)
    (hV : ∀ {n x}, x ∈ V n → x * x ∈ V n → x ∈ V (n + 1))
    (hVo : Filter.HasBasis (nhds 1) (fun _ ↦ True) V) :
    LocallyCompactSpace (ContinuousMonoidHom X Y) := by
  /-
    X : Type u_7
    Y : Type u_8
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : Group X
    inst✝⁶ : TopologicalGroup X
    inst✝⁵ : UniformSpace Y
    inst✝⁴ : CommGroup Y
    inst✝³ : UniformGroup Y
    inst✝² : T0Space Y
    inst✝¹ : CompactSpace Y
    inst✝ : LocallyCompactSpace X
    V : Nat → Set Y
    hV : ∀ {n : Nat} {x : Y}, Membership.mem (V n) x → Membership.mem (V n) (HMul. …
    hVo : (nhds 1).HasBasis (fun x => True) V
    ⊢ LocallyCompactSpace (ContinuousMonoidHom X Y)
  -/
  obtain ⟨U0, hU0c, hU0o⟩ := exists_compact_mem_nhds (1 : X)
  let U_aux : ℕ → {S : Set X | S ∈ nhds 1} :=
    Nat.rec ⟨U0, hU0o⟩ <| fun _ S ↦ let h := exists_closed_nhds_one_inv_eq_mul_subset S.2
      ⟨Classical.choose h, (Classical.choose_spec h).1⟩
  /-
    case intro.intro
    X : Type u_7
    Y : Type u_8
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : Group X
    inst✝⁶ : TopologicalGroup X
    inst✝⁵ : UniformSpace Y
    inst✝⁴ : CommGroup Y
    inst✝³ : UniformGroup Y
    inst✝² : T0Space Y
    inst✝¹ : CompactSpace Y
    inst✝ : LocallyCompactSpace X
    V : Nat → Set Y
    hV : ∀ {n : Nat} {x : Y}, Membership.mem (V n) x → Membership.mem (V n) (HMul. …
    hVo : (nhds 1).HasBasis (fun x => True) V
    U0 : Set X
    hU0c : IsCompact U0
    hU0o : Membership.mem (nhds 1) U0
    U_aux : Nat → ↑(setOf fun S => Membership.mem (nhds 1) S) :=
      fun t =>
        Nat.rec ⟨U0, hU0o⟩
          (fun x S =>
            let h := ⋯;
            ⟨Classical.choose h, ⋯⟩)
          t
    ⊢ LocallyCompactSpace (ContinuousMonoidHom X Y)
  -/
  let U : ℕ → Set X := fun n ↦ (U_aux n).1
  /-
    case intro.intro
    X : Type u_7
    Y : Type u_8
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : Group X
    inst✝⁶ : TopologicalGroup X
    inst✝⁵ : UniformSpace Y
    inst✝⁴ : CommGroup Y
    inst✝³ : UniformGroup Y
    inst✝² : T0Space Y
    inst✝¹ : CompactSpace Y
    inst✝ : LocallyCompactSpace X
    V : Nat → Set Y
    hV : ∀ {n : Nat} {x : Y}, Membership.mem (V n) x → Membership.mem (V n) (HMul. …
    hVo : (nhds 1).HasBasis (fun x => True) V
    U0 : Set X
    hU0c : IsCompact U0
    hU0o : Membership.mem (nhds 1) U0
    U_aux : Nat → ↑(setOf fun S => Membership.mem (nhds 1) S) :=
      fun t =>
        Nat.rec ⟨U0, hU0o⟩
          (fun x S =>
            let h := ⋯;
            ⟨Classical.choose h, ⋯⟩)
          t
    U : Nat → Set X := fun n => ↑(U_aux n)
    ⊢ LocallyCompactSpace (ContinuousMonoidHom X Y)
  -/
  have hU1 : ∀ n, U n ∈ nhds 1 := fun n ↦ (U_aux n).2
  have hU2 : ∀ n, U (n + 1) * U (n + 1) ⊆ U n :=
    fun n ↦ (Classical.choose_spec (exists_closed_nhds_one_inv_eq_mul_subset (U_aux n).2)).2.2.2
  have hU3 : ∀ n, U (n + 1) ⊆ U n :=
    fun n x hx ↦ hU2 n (mul_one x ▸ Set.mul_mem_mul hx (mem_of_mem_nhds (hU1 (n + 1))))
  have hU4 : ∀ f : X →* Y, Set.MapsTo f (U 0) (V 0) → ∀ n, Set.MapsTo f (U n) (V n) := by
    intro f hf n
    induction' n with n ih
    · exact hf
    · exact fun x hx ↦ hV (ih (hU3 n hx)) (map_mul f x x ▸ ih (hU2 n (Set.mul_mem_mul hx hx)))
  /-
    case intro.intro
    X : Type u_7
    Y : Type u_8
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : Group X
    inst✝⁶ : TopologicalGroup X
    inst✝⁵ : UniformSpace Y
    inst✝⁴ : CommGroup Y
    inst✝³ : UniformGroup Y
    inst✝² : T0Space Y
    inst✝¹ : CompactSpace Y
    inst✝ : LocallyCompactSpace X
    V : Nat → Set Y
    hV : ∀ {n : Nat} {x : Y}, Membership.mem (V n) x → Membership.mem (V n) (HMul. …
    hVo : (nhds 1).HasBasis (fun x => True) V
    U0 : Set X
    hU0c : IsCompact U0
    hU0o : Membership.mem (nhds 1) U0
    U_aux : Nat → ↑(setOf fun S => Membership.mem (nhds 1) S) :=
      fun t =>
        Nat.rec ⟨U0, hU0o⟩
          (fun x S =>
            let h := ⋯;
            ⟨Classical.choose h, ⋯⟩)
          t
    U : Nat → Set X := fun n => ↑(U_aux n)
    hU1 : ∀ (n : Nat), Membership.mem (nhds 1) (U n)
    hU2 : ∀ (n : Nat), HasSubset.Subset (HMul.hMul (U (HAdd.hAdd n 1)) (U (HAdd.hA …
    hU3 : ∀ (n : Nat), HasSubset.Subset (U (HAdd.hAdd n 1)) (U n)
    hU4 : ∀ (f : MonoidHom X Y), Set.MapsTo (⇑f) (U 0) (V 0) → ∀ (n : Nat), Set.Ma …
    ⊢ LocallyCompactSpace (ContinuousMonoidHom X Y)
  -/
  apply locallyCompactSpace_of_equicontinuousAt (U 0) (V 0) hU0c (hVo.mem_of_mem trivial)
  /-
    case intro.intro
    X : Type u_7
    Y : Type u_8
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : Group X
    inst✝⁶ : TopologicalGroup X
    inst✝⁵ : UniformSpace Y
    inst✝⁴ : CommGroup Y
    inst✝³ : UniformGroup Y
    inst✝² : T0Space Y
    inst✝¹ : CompactSpace Y
    inst✝ : LocallyCompactSpace X
    V : Nat → Set Y
    hV : ∀ {n : Nat} {x : Y}, Membership.mem (V n) x → Membership.mem (V n) (HMul. …
    hVo : (nhds 1).HasBasis (fun x => True) V
    U0 : Set X
    hU0c : IsCompact U0
    hU0o : Membership.mem (nhds 1) U0
    U_aux : Nat → ↑(setOf fun S => Membership.mem (nhds 1) S) :=
      fun t =>
        Nat.rec ⟨U0, hU0o⟩
          (fun x S =>
            let h := ⋯;
            ⟨Classical.choose h, ⋯⟩)
          t
    U : Nat → Set X := fun n => ↑(U_aux n)
    hU1 : ∀ (n : Nat), Membership.mem (nhds 1) (U n)
    hU2 : ∀ (n : Nat), HasSubset.Subset (HMul.hMul (U (HAdd.hAdd n 1)) (U (HAdd.hA …
    hU3 : ∀ (n : Nat), HasSubset.Subset (U (HAdd.hAdd n 1)) (U n)
    hU4 : ∀ (f : MonoidHom X Y), Set.MapsTo (⇑f) (U 0) (V 0) → ∀ (n : Nat), Set.Ma …
    ⊢ EquicontinuousAt (fun f => ⇑↑f) 1
  -/
  rw [hVo.uniformity_of_nhds_one.equicontinuousAt_iff_right]
  /-
    case intro.intro
    X : Type u_7
    Y : Type u_8
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : Group X
    inst✝⁶ : TopologicalGroup X
    inst✝⁵ : UniformSpace Y
    inst✝⁴ : CommGroup Y
    inst✝³ : UniformGroup Y
    inst✝² : T0Space Y
    inst✝¹ : CompactSpace Y
    inst✝ : LocallyCompactSpace X
    V : Nat → Set Y
    hV : ∀ {n : Nat} {x : Y}, Membership.mem (V n) x → Membership.mem (V n) (HMul. …
    hVo : (nhds 1).HasBasis (fun x => True) V
    U0 : Set X
    hU0c : IsCompact U0
    hU0o : Membership.mem (nhds 1) U0
    U_aux : Nat → ↑(setOf fun S => Membership.mem (nhds 1) S) :=
      fun t =>
        Nat.rec ⟨U0, hU0o⟩
          (fun x S =>
            let h := ⋯;
            ⟨Classical.choose h, ⋯⟩)
          t
    U : Nat → Set X := fun n => ↑(U_aux n)
    hU1 : ∀ (n : Nat), Membership.mem (nhds 1) (U n)
    hU2 : ∀ (n : Nat), HasSubset.Subset (HMul.hMul (U (HAdd.hAdd n 1)) (U (HAdd.hA …
    hU3 : ∀ (n : Nat), HasSubset.Subset (U (HAdd.hAdd n 1)) (U n)
    hU4 : ∀ (f : MonoidHom X Y), Set.MapsTo (⇑f) (U 0) (V 0) → ∀ (n : Nat), Set.Ma …
    ⊢ ∀ (k : Nat), True → Filter.Eventually (fun x => ∀ (i : ↑(setOf fun f => Set. …
  -/
  refine fun n _ ↦ Filter.eventually_iff_exists_mem.mpr ⟨U n, hU1 n, fun x hx ⟨f, hf⟩ ↦ ?_⟩
  /-
    case intro.intro
    X : Type u_7
    Y : Type u_8
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : Group X
    inst✝⁶ : TopologicalGroup X
    inst✝⁵ : UniformSpace Y
    inst✝⁴ : CommGroup Y
    inst✝³ : UniformGroup Y
    inst✝² : T0Space Y
    inst✝¹ : CompactSpace Y
    inst✝ : LocallyCompactSpace X
    V : Nat → Set Y
    hV : ∀ {n : Nat} {x : Y}, Membership.mem (V n) x → Membership.mem (V n) (HMul. …
    hVo : (nhds 1).HasBasis (fun x => True) V
    U0 : Set X
    hU0c : IsCompact U0
    hU0o : Membership.mem (nhds 1) U0
    U_aux : Nat → ↑(setOf fun S => Membership.mem (nhds 1) S) :=
      fun t =>
        Nat.rec ⟨U0, hU0o⟩
          (fun x S =>
            let h := ⋯;
            ⟨Classical.choose h, ⋯⟩)
          t
    U : Nat → Set X := fun n => ↑(U_aux n)
    hU1 : ∀ (n : Nat), Membership.mem (nhds 1) (U n)
    hU2 : ∀ (n : Nat), HasSubset.Subset (HMul.hMul (U (HAdd.hAdd n 1)) (U (HAdd.hA …
    hU3 : ∀ (n : Nat), HasSubset.Subset (U (HAdd.hAdd n 1)) (U n)
    hU4 : ∀ (f : MonoidHom X Y), Set.MapsTo (⇑f) (U 0) (V 0) → ∀ (n : Nat), Set.Ma …
    n : Nat
    x✝¹ : True
    x : X
    hx : Membership.mem (U n) x
    x✝ : ↑(setOf fun f => Set.MapsTo (⇑f) (U 0) (V 0))
    f : MonoidHom X Y
    hf : Membership.mem (setOf fun f => Set.MapsTo (⇑f) (U 0) (V 0)) f
    ⊢ Membership.mem (setOf fun x => Membership.mem (V n) (HDiv.hDiv x.2 x.1)) { f …
  -/
  rw [Set.mem_setOf_eq, map_one, div_one]
  /-
    case intro.intro
    X : Type u_7
    Y : Type u_8
    inst✝⁸ : TopologicalSpace X
    inst✝⁷ : Group X
    inst✝⁶ : TopologicalGroup X
    inst✝⁵ : UniformSpace Y
    inst✝⁴ : CommGroup Y
    inst✝³ : UniformGroup Y
    inst✝² : T0Space Y
    inst✝¹ : CompactSpace Y
    inst✝ : LocallyCompactSpace X
    V : Nat → Set Y
    hV : ∀ {n : Nat} {x : Y}, Membership.mem (V n) x → Membership.mem (V n) (HMul. …
    hVo : (nhds 1).HasBasis (fun x => True) V
    U0 : Set X
    hU0c : IsCompact U0
    hU0o : Membership.mem (nhds 1) U0
    U_aux : Nat → ↑(setOf fun S => Membership.mem (nhds 1) S) :=
      fun t =>
        Nat.rec ⟨U0, hU0o⟩
          (fun x S =>
            let h := ⋯;
            ⟨Classical.choose h, ⋯⟩)
          t
    U : Nat → Set X := fun n => ↑(U_aux n)
    hU1 : ∀ (n : Nat), Membership.mem (nhds 1) (U n)
    hU2 : ∀ (n : Nat), HasSubset.Subset (HMul.hMul (U (HAdd.hAdd n 1)) (U (HAdd.hA …
    hU3 : ∀ (n : Nat), HasSubset.Subset (U (HAdd.hAdd n 1)) (U n)
    hU4 : ∀ (f : MonoidHom X Y), Set.MapsTo (⇑f) (U 0) (V 0) → ∀ (n : Nat), Set.Ma …
    n : Nat
    x✝¹ : True
    x : X
    hx : Membership.mem (U n) x
    x✝ : ↑(setOf fun f => Set.MapsTo (⇑f) (U 0) (V 0))
    f : MonoidHom X Y
    hf : Membership.mem (setOf fun f => Set.MapsTo (⇑f) (U 0) (V 0)) f
    ⊢ Membership.mem (V n) { fst := 1, snd := ↑⟨f, hf⟩ x }.2
  -/
  exact hU4 f hf n hx
  /-
    🎉 no goals
  -/


/-- The structure of two-sided continuous isomorphisms between additive groups.
Note that both the map and its inverse have to be continuous. -/
structure ContinuousAddEquiv [Add G] [Add H] extends G ≃+ H , G ≃ₜ H


/-- The structure of two-sided continuous isomorphisms between groups.
Note that both the map and its inverse have to be continuous. -/
@[to_additive "The structure of two-sided continuous isomorphisms between additive groups.
Note that both the map and its inverse have to be continuous."]
structure ContinuousMulEquiv [Mul G] [Mul H] extends G ≃* H , G ≃ₜ H


@[inherit_doc]
infixl:25 " ≃ₜ* " => ContinuousMulEquiv


@[inherit_doc]
infixl:25 " ≃ₜ+ " => ContinuousAddEquiv


@[to_additive]
instance : EquivLike (M ≃ₜ* N) M N where
  coe f := f.toFun
  inv f := f.invFun
  left_inv f := f.left_inv
  right_inv f := f.right_inv
  coe_injective' f g h₁ h₂ := by
    /-
      G : Type u
      inst✝⁵ : TopologicalSpace G
      H : Type v
      inst✝⁴ : TopologicalSpace H
      M : Type u_1
      N : Type u_2
      inst✝³ : TopologicalSpace M
      inst✝² : TopologicalSpace N
      inst✝¹ : Mul M
      inst✝ : Mul N
      f g : ContinuousMulEquiv M N
      h₁ : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      h₂ : Eq ((fun f => f.invFun) f) ((fun f => f.invFun) g)
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      G : Type u
      inst✝⁵ : TopologicalSpace G
      H : Type v
      inst✝⁴ : TopologicalSpace H
      M : Type u_1
      N : Type u_2
      inst✝³ : TopologicalSpace M
      inst✝² : TopologicalSpace N
      inst✝¹ : Mul M
      inst✝ : Mul N
      g : ContinuousMulEquiv M N
      toMulEquiv✝ : MulEquiv M N
      continuous_toFun✝ : Continuous toMulEquiv✝.toFun
      continuous_invFun✝ : Continuous toMulEquiv✝.invFun
      h₁ : Eq ((fun f => f.toFun) { toMulEquiv := toMulEquiv✝, continuous_toFun := c …
      h₂ : Eq ((fun f => f.invFun) { toMulEquiv := toMulEquiv✝, continuous_toFun :=  …
      ⊢ Eq { toMulEquiv := toMulEquiv✝, continuous_toFun := continuous_toFun✝, conti …
    -/
    cases g
    /-
      case mk.mk
      G : Type u
      inst✝⁵ : TopologicalSpace G
      H : Type v
      inst✝⁴ : TopologicalSpace H
      M : Type u_1
      N : Type u_2
      inst✝³ : TopologicalSpace M
      inst✝² : TopologicalSpace N
      inst✝¹ : Mul M
      inst✝ : Mul N
      toMulEquiv✝¹ : MulEquiv M N
      continuous_toFun✝¹ : Continuous toMulEquiv✝¹.toFun
      continuous_invFun✝¹ : Continuous toMulEquiv✝¹.invFun
      toMulEquiv✝ : MulEquiv M N
      continuous_toFun✝ : Continuous toMulEquiv✝.toFun
      continuous_invFun✝ : Continuous toMulEquiv✝.invFun
      h₁ : Eq ((fun f => f.toFun) { toMulEquiv := toMulEquiv✝¹, continuous_toFun :=  …
      h₂ : Eq ((fun f => f.invFun) { toMulEquiv := toMulEquiv✝¹, continuous_toFun := …
      ⊢ Eq { toMulEquiv := toMulEquiv✝¹, continuous_toFun := continuous_toFun✝¹, con …
    -/
    congr
    /-
      case mk.mk.e_toMulEquiv
      G : Type u
      inst✝⁵ : TopologicalSpace G
      H : Type v
      inst✝⁴ : TopologicalSpace H
      M : Type u_1
      N : Type u_2
      inst✝³ : TopologicalSpace M
      inst✝² : TopologicalSpace N
      inst✝¹ : Mul M
      inst✝ : Mul N
      toMulEquiv✝¹ : MulEquiv M N
      continuous_toFun✝¹ : Continuous toMulEquiv✝¹.toFun
      continuous_invFun✝¹ : Continuous toMulEquiv✝¹.invFun
      toMulEquiv✝ : MulEquiv M N
      continuous_toFun✝ : Continuous toMulEquiv✝.toFun
      continuous_invFun✝ : Continuous toMulEquiv✝.invFun
      h₁ : Eq ((fun f => f.toFun) { toMulEquiv := toMulEquiv✝¹, continuous_toFun :=  …
      h₂ : Eq ((fun f => f.invFun) { toMulEquiv := toMulEquiv✝¹, continuous_toFun := …
      ⊢ Eq toMulEquiv✝¹ toMulEquiv✝
    -/
    exact MulEquiv.ext_iff.mpr (congrFun h₁)
    /-
      🎉 no goals
    -/


@[to_additive]
instance : MulEquivClass (M ≃ₜ* N) M N where
  map_mul f := f.map_mul'


@[to_additive]
instance : HomeomorphClass (M ≃ₜ* N) M N where
  map_continuous f := f.continuous_toFun
  inv_continuous f := f.continuous_invFun


/-- Two continuous multiplicative isomorphisms agree if they are defined by the
same underlying function. -/
@[to_additive (attr := ext)
  "Two continuous additive isomorphisms agree if they are defined by the same underlying function."]
theorem ext {f g : M ≃ₜ* N} (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext f g h


@[to_additive (attr := simp)]
theorem coe_mk (f : M ≃* N) (hf1 hf2) : ⇑(mk f hf1 hf2) = f := rfl


@[to_additive]
theorem toEquiv_eq_coe (f : M ≃ₜ* N) : f.toEquiv = f :=
  rfl


@[to_additive (attr := simp)]
theorem toMulEquiv_eq_coe (f : M ≃ₜ* N) : f.toMulEquiv = f :=
  rfl


@[to_additive]
theorem toHomeomorph_eq_coe (f : M ≃ₜ* N) : f.toHomeomorph = f :=
  rfl


/-- Makes a continuous multiplicative isomorphism from
a homeomorphism which preserves multiplication. -/
@[to_additive "Makes an continuous additive isomorphism from
a homeomorphism which preserves addition."]
def mk' (f : M ≃ₜ N) (h : ∀ x y, f (x * y) = f x * f y) : M ≃ₜ* N :=
  ⟨⟨f.toEquiv,h⟩, f.continuous_toFun, f.continuous_invFun⟩


set_option linter.docPrime false in -- This is about `ContinuousMulEquiv.mk'`
@[simp]
lemma coe_mk' (f : M ≃ₜ N) (h : ∀ x y, f (x * y) = f x * f y)  : ⇑(mk' f h) = f := rfl


@[to_additive]
protected theorem bijective (e : M ≃ₜ* N) : Function.Bijective e :=
  EquivLike.bijective e


@[to_additive]
protected theorem injective (e : M ≃ₜ* N) : Function.Injective e :=
  EquivLike.injective e


@[to_additive]
protected theorem surjective (e : M ≃ₜ* N) : Function.Surjective e :=
  EquivLike.surjective e


@[to_additive]
theorem apply_eq_iff_eq (e : M ≃ₜ* N) {x y : M} : e x = e y ↔ x = y :=
  e.injective.eq_iff


/-- The identity map is a continuous multiplicative isomorphism. -/
@[to_additive (attr := refl) "The identity map is a continuous additive isomorphism."]
def refl : M ≃ₜ* M :=
  { MulEquiv.refl _ with }


@[to_additive]
instance : Inhabited (M ≃ₜ* M) := ⟨ContinuousMulEquiv.refl M⟩


@[to_additive (attr := simp, norm_cast)]
theorem coe_refl : ↑(refl M) = id := rfl


@[to_additive (attr := simp)]
theorem refl_apply (m : M) : refl M m = m := rfl


/-- The inverse of a ContinuousMulEquiv. -/
@[to_additive (attr := symm) "The inverse of a ContinuousAddEquiv."]
def symm (cme : M ≃ₜ* N) : N ≃ₜ* M :=
  { cme.toMulEquiv.symm with
  continuous_toFun := cme.continuous_invFun
  continuous_invFun := cme.continuous_toFun }

@[to_additive]
theorem invFun_eq_symm {f : M ≃ₜ* N} : f.invFun = f.symm := rfl


@[to_additive (attr := simp)]
theorem coe_toHomeomorph_symm (f : M ≃ₜ* N) : (f : M ≃ₜ N).symm = (f.symm : N ≃ₜ M) := rfl


@[to_additive (attr := simp)]
theorem equivLike_inv_eq_symm (f : M ≃ₜ* N) : EquivLike.inv f = f.symm := rfl


@[to_additive (attr := simp)]
theorem symm_symm (f : M ≃ₜ* N) : f.symm.symm = f := rfl


/-- `e.symm` is a right inverse of `e`, written as `e (e.symm y) = y`. -/
@[to_additive (attr := simp) "`e.symm` is a right inverse of `e`, written as `e (e.symm y) = y`."]
theorem apply_symm_apply (e : M ≃ₜ* N) (y : N) : e (e.symm y) = y :=
  e.toEquiv.apply_symm_apply y


/-- `e.symm` is a left inverse of `e`, written as `e.symm (e y) = y`. -/
@[to_additive (attr := simp) "`e.symm` is a left inverse of `e`, written as `e.symm (e y) = y`."]
theorem symm_apply_apply (e : M ≃ₜ* N) (x : M) : e.symm (e x) = x :=
  e.toEquiv.symm_apply_apply x


@[to_additive (attr := simp)]
theorem symm_comp_self (e : M ≃ₜ* N) : e.symm ∘ e = id :=
  funext e.symm_apply_apply


@[to_additive (attr := simp)]
theorem self_comp_symm (e : M ≃ₜ* N) : e ∘ e.symm = id :=
  funext e.apply_symm_apply


@[to_additive]
theorem apply_eq_iff_symm_apply (e : M ≃ₜ* N) {x : M} {y : N} : e x = y ↔ x = e.symm y :=
  e.toEquiv.apply_eq_iff_eq_symm_apply


@[to_additive]
theorem symm_apply_eq (e : M ≃ₜ* N) {x y} : e.symm x = y ↔ x = e y :=
  e.toEquiv.symm_apply_eq


@[to_additive]
theorem eq_symm_apply (e : M ≃ₜ* N) {x y} : y = e.symm x ↔ e y = x :=
  e.toEquiv.eq_symm_apply


@[to_additive]
theorem eq_comp_symm {α : Type*} (e : M ≃ₜ* N) (f : N → α) (g : M → α) :
    f = g ∘ e.symm ↔ f ∘ e = g :=
  e.toEquiv.eq_comp_symm f g


@[to_additive]
theorem comp_symm_eq {α : Type*} (e : M ≃ₜ* N) (f : N → α) (g : M → α) :
    g ∘ e.symm = f ↔ g = f ∘ e :=
  e.toEquiv.comp_symm_eq f g


@[to_additive]
theorem eq_symm_comp {α : Type*} (e : M ≃ₜ* N) (f : α → M) (g : α → N) :
    f = e.symm ∘ g ↔ e ∘ f = g :=
  e.toEquiv.eq_symm_comp f g


@[to_additive]
theorem symm_comp_eq {α : Type*} (e : M ≃ₜ* N) (f : α → M) (g : α → N) :
    e.symm ∘ g = f ↔ g = e ∘ f :=
  e.toEquiv.symm_comp_eq f g


/-- The composition of two ContinuousMulEquiv. -/
@[to_additive "The composition of two ContinuousAddEquiv."]
def trans (cme1 : M ≃ₜ* N) (cme2 : N ≃ₜ* L) : M ≃ₜ* L :=
  { cme1.toMulEquiv.trans cme2.toMulEquiv with
                         /-
                           G : Type u
                           inst✝⁷ : TopologicalSpace G
                           H : Type v
                           inst✝⁶ : TopologicalSpace H
                           M : Type u_1
                           N : Type u_2
                           inst✝⁵ : TopologicalSpace M
                           inst✝⁴ : TopologicalSpace N
                           inst✝³ : Mul M
                           inst✝² : Mul N
                           L : Type u_3
                           inst✝¹ : Mul L
                           inst✝ : TopologicalSpace L
                           cme1 : ContinuousMulEquiv M N
                           cme2 : ContinuousMulEquiv N L
                           ⊢ Continuous __src✝.toFun
                         -/
  continuous_toFun := by convert Continuous.comp cme2.continuous_toFun cme1.continuous_toFun
                         /-
                           🎉 no goals
                         -/
                          /-
                            G : Type u
                            inst✝⁷ : TopologicalSpace G
                            H : Type v
                            inst✝⁶ : TopologicalSpace H
                            M : Type u_1
                            N : Type u_2
                            inst✝⁵ : TopologicalSpace M
                            inst✝⁴ : TopologicalSpace N
                            inst✝³ : Mul M
                            inst✝² : Mul N
                            L : Type u_3
                            inst✝¹ : Mul L
                            inst✝ : TopologicalSpace L
                            cme1 : ContinuousMulEquiv M N
                            cme2 : ContinuousMulEquiv N L
                            ⊢ Continuous __src✝.invFun
                          -/
  continuous_invFun := by convert Continuous.comp cme1.continuous_invFun cme2.continuous_invFun }
                          /-
                            🎉 no goals
                          -/


@[to_additive (attr := simp)]
theorem coe_trans (e₁ : M ≃ₜ* N) (e₂ : N ≃ₜ* L) : ↑(e₁.trans e₂) = e₂ ∘ e₁ := rfl


@[to_additive (attr := simp)]
theorem trans_apply (e₁ : M ≃ₜ* N) (e₂ : N ≃ₜ* L) (m : M) : e₁.trans e₂ m = e₂ (e₁ m) := rfl


@[to_additive (attr := simp)]
theorem symm_trans_apply (e₁ : M ≃ₜ* N) (e₂ : N ≃ₜ* L) (l : L) :
    (e₁.trans e₂).symm l = e₁.symm (e₂.symm l) := rfl


@[to_additive (attr := simp)]
theorem symm_trans_self (e : M ≃ₜ* N) : e.symm.trans e = refl N :=
  DFunLike.ext _ _ e.apply_symm_apply


@[to_additive (attr := simp)]
theorem self_trans_symm (e : M ≃ₜ* N) : e.trans e.symm = refl M :=
  DFunLike.ext _ _ e.symm_apply_apply


/-- The `MulEquiv` between two monoids with a unique element. -/
@[to_additive "The `AddEquiv` between two `AddMonoid`s with a unique element."]
def ofUnique {M N} [Unique M] [Unique N] [Mul M] [Mul N]
    [TopologicalSpace M] [TopologicalSpace N] : M ≃ₜ* N :=
  { MulEquiv.ofUnique with
                         /-
                           G : Type u
                           inst✝¹¹ : TopologicalSpace G
                           H : Type v
                           inst✝¹⁰ : TopologicalSpace H
                           M✝ : Type u_1
                           N✝ : Type u_2
                           inst✝⁹ : TopologicalSpace M✝
                           inst✝⁸ : TopologicalSpace N✝
                           inst✝⁷ : Mul M✝
                           inst✝⁶ : Mul N✝
                           M : Type ?u.125314
                           N : Type ?u.125317
                           inst✝⁵ : Unique M
                           inst✝⁴ : Unique N
                           inst✝³ : Mul M
                           inst✝² : Mul N
                           inst✝¹ : TopologicalSpace M
                           inst✝ : TopologicalSpace N
                           ⊢ Continuous __src✝.toFun
                         -/
  continuous_toFun := by continuity
                         /-
                           🎉 no goals
                         -/
                          /-
                            G : Type u
                            inst✝¹¹ : TopologicalSpace G
                            H : Type v
                            inst✝¹⁰ : TopologicalSpace H
                            M✝ : Type u_1
                            N✝ : Type u_2
                            inst✝⁹ : TopologicalSpace M✝
                            inst✝⁸ : TopologicalSpace N✝
                            inst✝⁷ : Mul M✝
                            inst✝⁶ : Mul N✝
                            M : Type ?u.125314
                            N : Type ?u.125317
                            inst✝⁵ : Unique M
                            inst✝⁴ : Unique N
                            inst✝³ : Mul M
                            inst✝² : Mul N
                            inst✝¹ : TopologicalSpace M
                            inst✝ : TopologicalSpace N
                            ⊢ Continuous __src✝.invFun
                          -/
  continuous_invFun := by continuity }
                          /-
                            🎉 no goals
                          -/


/-- There is a unique monoid homomorphism between two monoids with a unique element. -/
@[to_additive "There is a unique additive monoid homomorphism between two additive monoids with
  a unique element."]
instance {M N} [Unique M] [Unique N] [Mul M] [Mul N]
    [TopologicalSpace M] [TopologicalSpace N] : Unique (M ≃ₜ* N) where
  default := ofUnique
  uniq _ := ext fun _ ↦ Subsingleton.elim _ _



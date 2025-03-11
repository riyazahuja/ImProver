/-- A continuous alternating map from `ι → M` to `N`, denoted `M [⋀^ι]→L[R] N`,
is a continuous map that is

- multilinear : `f (update m i (c • x)) = c • f (update m i x)` and
  `f (update m i (x + y)) = f (update m i x) + f (update m i y)`;
- alternating : `f v = 0` whenever `v` has two equal coordinates.
-/
structure ContinuousAlternatingMap (R M N ι : Type*) [Semiring R] [AddCommMonoid M] [Module R M]
    [TopologicalSpace M] [AddCommMonoid N] [Module R N] [TopologicalSpace N] extends
    ContinuousMultilinearMap R (fun _ : ι => M) N, M [⋀^ι]→ₗ[R] N where


@[inherit_doc]
notation M " [⋀^" ι "]→L[" R "] " N:100 => ContinuousAlternatingMap R M N ι


theorem toContinuousMultilinearMap_injective :
    Injective (ContinuousAlternatingMap.toContinuousMultilinearMap :
      M [⋀^ι]→L[R] N → ContinuousMultilinearMap R (fun _ : ι => M) N)
  | ⟨_, _⟩, ⟨_, _⟩, rfl => rfl


theorem range_toContinuousMultilinearMap :
    Set.range
        (toContinuousMultilinearMap :
          M [⋀^ι]→L[R] N → ContinuousMultilinearMap R (fun _ : ι => M) N) =
      {f | ∀ (v : ι → M) (i j : ι), v i = v j → i ≠ j → f v = 0} :=
  Set.ext fun f => ⟨fun ⟨g, hg⟩ => hg ▸ g.2, fun h => ⟨⟨f, h⟩, rfl⟩⟩


instance funLike : FunLike (M [⋀^ι]→L[R] N) (ι → M) N where
  coe f := f.toFun
  coe_injective' _ _ h := toContinuousMultilinearMap_injective <| DFunLike.ext' h


instance continuousMapClass : ContinuousMapClass (M [⋀^ι]→L[R] N) (ι → M) N where
  map_continuous f := f.cont


@[continuity]
theorem coe_continuous : Continuous f := f.cont


@[simp]
theorem coe_toContinuousMultilinearMap : ⇑f.toContinuousMultilinearMap = f :=
  rfl


@[simp]
theorem coe_mk (f : ContinuousMultilinearMap R (fun _ : ι => M) N) (h) : ⇑(mk f h) = f :=
  rfl

-- not a `simp` lemma because this projection is a reducible call to `mk`, so `simp` can prove
-- this lemma

theorem coe_toAlternatingMap : ⇑f.toAlternatingMap = f := rfl


@[ext]
theorem ext {f g : M [⋀^ι]→L[R] N} (H : ∀ x, f x = g x) : f = g :=
  DFunLike.ext _ _ H


theorem toAlternatingMap_injective :
    Injective (toAlternatingMap : (M [⋀^ι]→L[R] N) → (M [⋀^ι]→ₗ[R] N)) := fun f g h =>
                      /-
                        R : Type u_1
                        M : Type u_2
                        N : Type u_4
                        ι : Type u_6
                        inst✝⁶ : Semiring R
                        inst✝⁵ : AddCommMonoid M
                        inst✝⁴ : Module R M
                        inst✝³ : TopologicalSpace M
                        inst✝² : AddCommMonoid N
                        inst✝¹ : Module R N
                        inst✝ : TopologicalSpace N
                        f g : ContinuousAlternatingMap R M N ι
                        h : Eq f.toAlternatingMap g.toAlternatingMap
                        ⊢ Eq ⇑f ⇑g
                      -/
  DFunLike.ext' <| by convert DFunLike.ext'_iff.1 h
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem range_toAlternatingMap :
    Set.range (toAlternatingMap : M [⋀^ι]→L[R] N → (M [⋀^ι]→ₗ[R] N)) =
      {f : M [⋀^ι]→ₗ[R] N | Continuous f} :=
  Set.ext fun f => ⟨fun ⟨g, hg⟩ => hg ▸ g.cont, fun h => ⟨{ f with cont := h }, DFunLike.ext' rfl⟩⟩


@[simp]
theorem map_update_add [DecidableEq ι] (m : ι → M) (i : ι) (x y : M) :
    f (update m i (x + y)) = f (update m i x) + f (update m i y) :=
  f.map_update_add' m i x y


@[deprecated (since := "2024-11-03")] protected alias map_add := map_update_add


@[simp]
theorem map_update_smul [DecidableEq ι] (m : ι → M) (i : ι) (c : R) (x : M) :
    f (update m i (c • x)) = c • f (update m i x) :=
  f.map_update_smul' m i c x


@[deprecated (since := "2024-11-03")] protected alias map_smul := map_update_smul


theorem map_coord_zero {m : ι → M} (i : ι) (h : m i = 0) : f m = 0 :=
  f.toMultilinearMap.map_coord_zero i h


@[simp]
theorem map_update_zero [DecidableEq ι] (m : ι → M) (i : ι) : f (update m i 0) = 0 :=
  f.toMultilinearMap.map_update_zero m i


@[simp]
theorem map_zero [Nonempty ι] : f 0 = 0 :=
  f.toMultilinearMap.map_zero


theorem map_eq_zero_of_eq (v : ι → M) {i j : ι} (h : v i = v j) (hij : i ≠ j) : f v = 0 :=
  f.map_eq_zero_of_eq' v i j h hij


theorem map_eq_zero_of_not_injective (v : ι → M) (hv : ¬Function.Injective v) : f v = 0 :=
  f.toAlternatingMap.map_eq_zero_of_not_injective v hv


/-- Restrict the codomain of a continuous alternating map to a submodule. -/
@[simps!]
def codRestrict (f : M [⋀^ι]→L[R] N) (p : Submodule R N) (h : ∀ v, f v ∈ p) : M [⋀^ι]→L[R] p :=
  { f.toAlternatingMap.codRestrict p h with toContinuousMultilinearMap := f.1.codRestrict p h }


instance : Zero (M [⋀^ι]→L[R] N) :=
  ⟨⟨0, (0 : M [⋀^ι]→ₗ[R] N).map_eq_zero_of_eq⟩⟩


instance : Inhabited (M [⋀^ι]→L[R] N) :=
  ⟨0⟩


@[simp]
theorem coe_zero : ⇑(0 : M [⋀^ι]→L[R] N) = 0 :=
  rfl


@[simp]
theorem toContinuousMultilinearMap_zero : (0 : M [⋀^ι]→L[R] N).toContinuousMultilinearMap = 0 :=
  rfl


@[simp]
theorem toAlternatingMap_zero : (0 : M [⋀^ι]→L[R] N).toAlternatingMap = 0 :=
  rfl


instance : SMul R' (M [⋀^ι]→L[A] N) :=
  ⟨fun c f => ⟨c • f.1, (c • f.toAlternatingMap).map_eq_zero_of_eq⟩⟩


@[simp]
theorem coe_smul (f : M [⋀^ι]→L[A] N) (c : R') : ⇑(c • f) = c • ⇑f :=
  rfl


theorem smul_apply (f : M [⋀^ι]→L[A] N) (c : R') (v : ι → M) : (c • f) v = c • f v :=
  rfl


@[simp]
theorem toContinuousMultilinearMap_smul (c : R') (f : M [⋀^ι]→L[A] N) :
    (c • f).toContinuousMultilinearMap = c • f.toContinuousMultilinearMap :=
  rfl


@[simp]
theorem toAlternatingMap_smul (c : R') (f : M [⋀^ι]→L[A] N) :
    (c • f).toAlternatingMap = c • f.toAlternatingMap :=
  rfl


instance [SMulCommClass R' R'' N] : SMulCommClass R' R'' (M [⋀^ι]→L[A] N) :=
  ⟨fun _ _ _ => ext fun _ => smul_comm _ _ _⟩


instance [SMul R' R''] [IsScalarTower R' R'' N] : IsScalarTower R' R'' (M [⋀^ι]→L[A] N) :=
  ⟨fun _ _ _ => ext fun _ => smul_assoc _ _ _⟩


instance [DistribMulAction R'ᵐᵒᵖ N] [IsCentralScalar R' N] : IsCentralScalar R' (M [⋀^ι]→L[A] N) :=
  ⟨fun _ _ => ext fun _ => op_smul_eq_smul _ _⟩


instance : MulAction R' (M [⋀^ι]→L[A] N) :=
  toContinuousMultilinearMap_injective.mulAction toContinuousMultilinearMap fun _ _ => rfl


instance : Add (M [⋀^ι]→L[R] N) :=
  ⟨fun f g => ⟨f.1 + g.1, (f.toAlternatingMap + g.toAlternatingMap).map_eq_zero_of_eq⟩⟩


@[simp]
theorem coe_add : ⇑(f + g) = ⇑f + ⇑g :=
  rfl


@[simp]
theorem add_apply (v : ι → M) : (f + g) v = f v + g v :=
  rfl


@[simp]
theorem toContinuousMultilinearMap_add (f g : M [⋀^ι]→L[R] N) : (f + g).1 = f.1 + g.1 :=
  rfl


@[simp]
theorem toAlternatingMap_add (f g : M [⋀^ι]→L[R] N) :
    (f + g).toAlternatingMap = f.toAlternatingMap + g.toAlternatingMap :=
  rfl


instance addCommMonoid : AddCommMonoid (M [⋀^ι]→L[R] N) :=
  toContinuousMultilinearMap_injective.addCommMonoid _ rfl (fun _ _ => rfl) fun _ _ => rfl


/-- Evaluation of a `ContinuousAlternatingMap` at a vector as an `AddMonoidHom`. -/
def applyAddHom (v : ι → M) : M [⋀^ι]→L[R] N →+ N :=
  ⟨⟨fun f => f v, rfl⟩, fun _ _ => rfl⟩


@[simp]
theorem sum_apply {α : Type*} (f : α → M [⋀^ι]→L[R] N) (m : ι → M) {s : Finset α} :
    (∑ a ∈ s, f a) m = ∑ a ∈ s, f a m :=
  map_sum (applyAddHom m) f s


/-- Projection to `ContinuousMultilinearMap`s as a bundled `AddMonoidHom`. -/
@[simps]
def toMultilinearAddHom : M [⋀^ι]→L[R] N →+ ContinuousMultilinearMap R (fun _ : ι => M) N :=
  ⟨⟨fun f => f.1, rfl⟩, fun _ _ => rfl⟩


/-- If `f` is a continuous alternating map, then `f.toContinuousLinearMap m i` is the continuous
linear map obtained by fixing all coordinates but `i` equal to those of `m`, and varying the
`i`-th coordinate. -/
@[simps! apply]
def toContinuousLinearMap [DecidableEq ι] (m : ι → M) (i : ι) : M →L[R] N :=
  f.1.toContinuousLinearMap m i


/-- The cartesian product of two continuous alternating maps, as a continuous alternating map. -/
@[simps!]
def prod (f : M [⋀^ι]→L[R] N) (g : M [⋀^ι]→L[R] N') : M [⋀^ι]→L[R] (N × N') :=
  ⟨f.1.prod g.1, (f.toAlternatingMap.prod g.toAlternatingMap).map_eq_zero_of_eq⟩


/-- Combine a family of continuous alternating maps with the same domain and codomains `M' i` into a
continuous alternating map taking values in the space of functions `Π i, M' i`. -/
def pi {ι' : Type*} {M' : ι' → Type*} [∀ i, AddCommMonoid (M' i)] [∀ i, TopologicalSpace (M' i)]
    [∀ i, Module R (M' i)] (f : ∀ i, M [⋀^ι]→L[R] M' i) : M [⋀^ι]→L[R] ∀ i, M' i :=
  ⟨ContinuousMultilinearMap.pi fun i => (f i).1,
    (AlternatingMap.pi fun i => (f i).toAlternatingMap).map_eq_zero_of_eq⟩


@[simp]
theorem coe_pi {ι' : Type*} {M' : ι' → Type*} [∀ i, AddCommMonoid (M' i)]
    [∀ i, TopologicalSpace (M' i)] [∀ i, Module R (M' i)] (f : ∀ i, M [⋀^ι]→L[R] M' i) :
    ⇑(pi f) = fun m j => f j m :=
  rfl


theorem pi_apply {ι' : Type*} {M' : ι' → Type*} [∀ i, AddCommMonoid (M' i)]
    [∀ i, TopologicalSpace (M' i)] [∀ i, Module R (M' i)] (f : ∀ i, M [⋀^ι]→L[R] M' i) (m : ι → M)
    (j : ι') : pi f m j = f j m :=
  rfl


/-- The natural equivalence between continuous linear maps from `M` to `N`
and continuous 1-multilinear alternating maps from `M` to `N`. -/
@[simps! apply_apply symm_apply_apply apply_toContinuousMultilinearMap]
def ofSubsingleton [Subsingleton ι] (i : ι) :
    (M →L[R] N) ≃ M [⋀^ι]→L[R] N where
  toFun f :=
    { AlternatingMap.ofSubsingleton R M N i f with
      toContinuousMultilinearMap := ContinuousMultilinearMap.ofSubsingleton R M N i f }
  invFun f := (ContinuousMultilinearMap.ofSubsingleton R M N i).symm f.1
  left_inv _ := rfl
  right_inv _ := toContinuousMultilinearMap_injective <|
    (ContinuousMultilinearMap.ofSubsingleton R M N i).apply_symm_apply _


@[simp]
theorem ofSubsingleton_toAlternatingMap [Subsingleton ι] (i : ι) (f : M →L[R] N) :
    (ofSubsingleton R M N i f).toAlternatingMap = AlternatingMap.ofSubsingleton R M N i f :=
  rfl


/-- The constant map is alternating when `ι` is empty. -/
@[simps! toContinuousMultilinearMap apply]
def constOfIsEmpty [IsEmpty ι] (m : N) : M [⋀^ι]→L[R] N :=
  { AlternatingMap.constOfIsEmpty R M ι m with
    toContinuousMultilinearMap := ContinuousMultilinearMap.constOfIsEmpty R (fun _ => M) m }


@[simp]
theorem constOfIsEmpty_toAlternatingMap [IsEmpty ι] (m : N) :
    (constOfIsEmpty R M ι m).toAlternatingMap = AlternatingMap.constOfIsEmpty R M ι m :=
  rfl


/-- If `g` is continuous alternating and `f` is a continuous linear map, then `g (f m₁, ..., f mₙ)`
is again a continuous alternating map, that we call `g.compContinuousLinearMap f`. -/
def compContinuousLinearMap (g : M [⋀^ι]→L[R] N) (f : M' →L[R] M) : M' [⋀^ι]→L[R] N :=
  { g.toAlternatingMap.compLinearMap (f : M' →ₗ[R] M) with
    toContinuousMultilinearMap := g.1.compContinuousLinearMap fun _ => f }


@[simp]
theorem compContinuousLinearMap_apply (g : M [⋀^ι]→L[R] N) (f : M' →L[R] M) (m : ι → M') :
    g.compContinuousLinearMap f m = g (f ∘ m) :=
  rfl


/-- Composing a continuous alternating map with a continuous linear map gives again a
continuous alternating map. -/
def _root_.ContinuousLinearMap.compContinuousAlternatingMap (g : N →L[R] N') (f : M [⋀^ι]→L[R] N) :
    M [⋀^ι]→L[R] N' :=
  { (g : N →ₗ[R] N').compAlternatingMap f.toAlternatingMap with
    toContinuousMultilinearMap := g.compContinuousMultilinearMap f.1 }


@[simp]
theorem _root_.ContinuousLinearMap.compContinuousAlternatingMap_coe (g : N →L[R] N')
    (f : M [⋀^ι]→L[R] N) : ⇑(g.compContinuousAlternatingMap f) = g ∘ f :=
  rfl


/-- A continuous linear equivalence of domains defines an equivalence between continuous
alternating maps. -/
def _root_.ContinuousLinearEquiv.continuousAlternatingMapComp (e : M ≃L[R] M') :
    M [⋀^ι]→L[R] N ≃ M' [⋀^ι]→L[R] N where
  toFun f := f.compContinuousLinearMap ↑e.symm
  invFun f := f.compContinuousLinearMap ↑e
                   /-
                     R : Type u_1
                     M : Type u_2
                     M' : Type u_3
                     N : Type u_4
                     N' : Type u_5
                     ι : Type u_6
                     inst✝¹² : Semiring R
                     inst✝¹¹ : AddCommMonoid M
                     inst✝¹⁰ : Module R M
                     inst✝⁹ : TopologicalSpace M
                     inst✝⁸ : AddCommMonoid M'
                     inst✝⁷ : Module R M'
                     inst✝⁶ : TopologicalSpace M'
                     inst✝⁵ : AddCommMonoid N
                     inst✝⁴ : Module R N
                     inst✝³ : TopologicalSpace N
                     inst✝² : AddCommMonoid N'
                     inst✝¹ : Module R N'
                     inst✝ : TopologicalSpace N'
                     n : Nat
                     f✝ g : ContinuousAlternatingMap R M N ι
                     e : ContinuousLinearEquiv (RingHom.id R) M M'
                     f : ContinuousAlternatingMap R M N ι
                     ⊢ Eq ((fun f => f.compContinuousLinearMap ↑e) ((fun f => f.compContinuousLinea …
                   -/
  left_inv f := by ext; simp [Function.comp_def]
                        /-
                          🎉 no goals
                        -/
                    /-
                      R : Type u_1
                      M : Type u_2
                      M' : Type u_3
                      N : Type u_4
                      N' : Type u_5
                      ι : Type u_6
                      inst✝¹² : Semiring R
                      inst✝¹¹ : AddCommMonoid M
                      inst✝¹⁰ : Module R M
                      inst✝⁹ : TopologicalSpace M
                      inst✝⁸ : AddCommMonoid M'
                      inst✝⁷ : Module R M'
                      inst✝⁶ : TopologicalSpace M'
                      inst✝⁵ : AddCommMonoid N
                      inst✝⁴ : Module R N
                      inst✝³ : TopologicalSpace N
                      inst✝² : AddCommMonoid N'
                      inst✝¹ : Module R N'
                      inst✝ : TopologicalSpace N'
                      n : Nat
                      f✝ g : ContinuousAlternatingMap R M N ι
                      e : ContinuousLinearEquiv (RingHom.id R) M M'
                      f : ContinuousAlternatingMap R M' N ι
                      ⊢ Eq ((fun f => f.compContinuousLinearMap ↑e.symm) ((fun f => f.compContinuous …
                    -/
  right_inv f := by ext; simp [Function.comp_def]
                         /-
                           🎉 no goals
                         -/


/-- A continuous linear equivalence of codomains defines an equivalence between continuous
alternating maps. -/
def _root_.ContinuousLinearEquiv.compContinuousAlternatingMap (e : N ≃L[R] N') :
    M [⋀^ι]→L[R] N ≃ M [⋀^ι]→L[R] N' where
  toFun := (e : N →L[R] N').compContinuousAlternatingMap
  invFun := (e.symm : N' →L[R] N).compContinuousAlternatingMap
                   /-
                     R : Type u_1
                     M : Type u_2
                     M' : Type u_3
                     N : Type u_4
                     N' : Type u_5
                     ι : Type u_6
                     inst✝¹² : Semiring R
                     inst✝¹¹ : AddCommMonoid M
                     inst✝¹⁰ : Module R M
                     inst✝⁹ : TopologicalSpace M
                     inst✝⁸ : AddCommMonoid M'
                     inst✝⁷ : Module R M'
                     inst✝⁶ : TopologicalSpace M'
                     inst✝⁵ : AddCommMonoid N
                     inst✝⁴ : Module R N
                     inst✝³ : TopologicalSpace N
                     inst✝² : AddCommMonoid N'
                     inst✝¹ : Module R N'
                     inst✝ : TopologicalSpace N'
                     n : Nat
                     f✝ g : ContinuousAlternatingMap R M N ι
                     e : ContinuousLinearEquiv (RingHom.id R) N N'
                     f : ContinuousAlternatingMap R M N ι
                     ⊢ Eq ((↑e.symm).compContinuousAlternatingMap ((↑e).compContinuousAlternatingMa …
                   -/
  left_inv f := by ext; simp [(· ∘ ·)]
                        /-
                          🎉 no goals
                        -/
                    /-
                      R : Type u_1
                      M : Type u_2
                      M' : Type u_3
                      N : Type u_4
                      N' : Type u_5
                      ι : Type u_6
                      inst✝¹² : Semiring R
                      inst✝¹¹ : AddCommMonoid M
                      inst✝¹⁰ : Module R M
                      inst✝⁹ : TopologicalSpace M
                      inst✝⁸ : AddCommMonoid M'
                      inst✝⁷ : Module R M'
                      inst✝⁶ : TopologicalSpace M'
                      inst✝⁵ : AddCommMonoid N
                      inst✝⁴ : Module R N
                      inst✝³ : TopologicalSpace N
                      inst✝² : AddCommMonoid N'
                      inst✝¹ : Module R N'
                      inst✝ : TopologicalSpace N'
                      n : Nat
                      f✝ g : ContinuousAlternatingMap R M N ι
                      e : ContinuousLinearEquiv (RingHom.id R) N N'
                      f : ContinuousAlternatingMap R M N' ι
                      ⊢ Eq ((↑e).compContinuousAlternatingMap ((↑e.symm).compContinuousAlternatingMa …
                    -/
  right_inv f := by ext; simp [(· ∘ ·)]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem _root_.ContinuousLinearEquiv.compContinuousAlternatingMap_coe
    (e : N ≃L[R] N') (f : M [⋀^ι]→L[R] N) : ⇑(e.compContinuousAlternatingMap f) = e ∘ f :=
  rfl


/-- Continuous linear equivalences between domains and codomains define an equivalence between the
spaces of continuous alternating maps. -/
def _root_.ContinuousLinearEquiv.continuousAlternatingMapCongr (e : M ≃L[R] M') (e' : N ≃L[R] N') :
    M [⋀^ι]→L[R] N ≃ M' [⋀^ι]→L[R] N' :=
  e.continuousAlternatingMapComp.trans e'.compContinuousAlternatingMap


/-- `ContinuousAlternatingMap.pi` as an `Equiv`. -/
@[simps]
def piEquiv {ι' : Type*} {N : ι' → Type*} [∀ i, AddCommMonoid (N i)] [∀ i, TopologicalSpace (N i)]
    [∀ i, Module R (N i)] : (∀ i, M [⋀^ι]→L[R] N i) ≃ M [⋀^ι]→L[R] ∀ i, N i where
  toFun := pi
  invFun f i := (ContinuousLinearMap.proj i : _ →L[R] N i).compContinuousAlternatingMap f
                   /-
                     R : Type u_1
                     M : Type u_2
                     M' : Type u_3
                     N✝ : Type u_4
                     N' : Type u_5
                     ι : Type u_6
                     inst✝¹⁵ : Semiring R
                     inst✝¹⁴ : AddCommMonoid M
                     inst✝¹³ : Module R M
                     inst✝¹² : TopologicalSpace M
                     inst✝¹¹ : AddCommMonoid M'
                     inst✝¹⁰ : Module R M'
                     inst✝⁹ : TopologicalSpace M'
                     inst✝⁸ : AddCommMonoid N✝
                     inst✝⁷ : Module R N✝
                     inst✝⁶ : TopologicalSpace N✝
                     inst✝⁵ : AddCommMonoid N'
                     inst✝⁴ : Module R N'
                     inst✝³ : TopologicalSpace N'
                     n : Nat
                     f✝ g : ContinuousAlternatingMap R M N✝ ι
                     ι' : Type u_7
                     N : ι' → Type u_8
                     inst✝² : (i : ι') → AddCommMonoid (N i)
                     inst✝¹ : (i : ι') → TopologicalSpace (N i)
                     inst✝ : (i : ι') → Module R (N i)
                     f : (i : ι') → ContinuousAlternatingMap R M (N i) ι
                     ⊢ Eq ((fun f i => (ContinuousLinearMap.proj i).compContinuousAlternatingMap f) …
                   -/
  left_inv f := by ext; rfl
                        /-
                          🎉 no goals
                        -/
                    /-
                      R : Type u_1
                      M : Type u_2
                      M' : Type u_3
                      N✝ : Type u_4
                      N' : Type u_5
                      ι : Type u_6
                      inst✝¹⁵ : Semiring R
                      inst✝¹⁴ : AddCommMonoid M
                      inst✝¹³ : Module R M
                      inst✝¹² : TopologicalSpace M
                      inst✝¹¹ : AddCommMonoid M'
                      inst✝¹⁰ : Module R M'
                      inst✝⁹ : TopologicalSpace M'
                      inst✝⁸ : AddCommMonoid N✝
                      inst✝⁷ : Module R N✝
                      inst✝⁶ : TopologicalSpace N✝
                      inst✝⁵ : AddCommMonoid N'
                      inst✝⁴ : Module R N'
                      inst✝³ : TopologicalSpace N'
                      n : Nat
                      f✝ g : ContinuousAlternatingMap R M N✝ ι
                      ι' : Type u_7
                      N : ι' → Type u_8
                      inst✝² : (i : ι') → AddCommMonoid (N i)
                      inst✝¹ : (i : ι') → TopologicalSpace (N i)
                      inst✝ : (i : ι') → Module R (N i)
                      f : ContinuousAlternatingMap R M ((i : ι') → N i) ι
                      ⊢ Eq (ContinuousAlternatingMap.pi ((fun f i => (ContinuousLinearMap.proj i).co …
                    -/
  right_inv f := by ext; rfl
                         /-
                           🎉 no goals
                         -/


/-- In the specific case of continuous alternating maps on spaces indexed by `Fin (n+1)`, where one
can build an element of `Π(i : Fin (n+1)), M i` using `cons`, one can express directly the
additivity of an alternating map along the first variable. -/
theorem cons_add (f : ContinuousAlternatingMap R M N (Fin (n + 1))) (m : Fin n → M) (x y : M) :
    f (Fin.cons (x + y) m) = f (Fin.cons x m) + f (Fin.cons y m) :=
  f.toMultilinearMap.cons_add m x y


/-- In the specific case of continuous alternating maps on spaces indexed by `Fin (n+1)`, where one
can build an element of `Π(i : Fin (n+1)), M i` using `cons`, one can express directly the
additivity of an alternating map along the first variable. -/
theorem vecCons_add (f : ContinuousAlternatingMap R M N (Fin (n + 1))) (m : Fin n → M) (x y : M) :
    f (vecCons (x + y) m) = f (vecCons x m) + f (vecCons y m) :=
  f.toMultilinearMap.cons_add m x y


/-- In the specific case of continuous alternating maps on spaces indexed by `Fin (n+1)`, where one
can build an element of `Π(i : Fin (n+1)), M i` using `cons`, one can express directly the
multiplicativity of an alternating map along the first variable. -/
theorem cons_smul (f : ContinuousAlternatingMap R M N (Fin (n + 1))) (m : Fin n → M) (c : R)
    (x : M) : f (Fin.cons (c • x) m) = c • f (Fin.cons x m) :=
  f.toMultilinearMap.cons_smul m c x


/-- In the specific case of continuous alternating maps on spaces indexed by `Fin (n+1)`, where one
can build an element of `Π(i : Fin (n+1)), M i` using `cons`, one can express directly the
multiplicativity of an alternating map along the first variable. -/
theorem vecCons_smul (f : ContinuousAlternatingMap R M N (Fin (n + 1))) (m : Fin n → M) (c : R)
    (x : M) : f (vecCons (c • x) m) = c • f (vecCons x m) :=
  f.toMultilinearMap.cons_smul m c x


theorem map_piecewise_add [DecidableEq ι] (m m' : ι → M) (t : Finset ι) :
    f (t.piecewise (m + m') m') = ∑ s ∈ t.powerset, f (s.piecewise m m') :=
  f.toMultilinearMap.map_piecewise_add _ _ _


/-- Additivity of a continuous alternating map along all coordinates at the same time,
writing `f (m + m')` as the sum of `f (s.piecewise m m')` over all sets `s`. -/
theorem map_add_univ [DecidableEq ι] [Fintype ι] (m m' : ι → M) :
    f (m + m') = ∑ s : Finset ι, f (s.piecewise m m') :=
  f.toMultilinearMap.map_add_univ _ _


/-- If `f` is continuous alternating, then `f (Σ_{j₁ ∈ A₁} g₁ j₁, ..., Σ_{jₙ ∈ Aₙ} gₙ jₙ)` is the
sum of `f (g₁ (r 1), ..., gₙ (r n))` where `r` ranges over all functions with `r 1 ∈ A₁`, ...,
`r n ∈ Aₙ`. This follows from multilinearity by expanding successively with respect to each
coordinate. -/
theorem map_sum_finset :
    (f fun i => ∑ j ∈ A i, g' i j) = ∑ r ∈ piFinset A, f fun i => g' i (r i) :=
  f.toMultilinearMap.map_sum_finset _ _


/-- If `f` is continuous alternating, then `f (Σ_{j₁} g₁ j₁, ..., Σ_{jₙ} gₙ jₙ)` is the sum of
`f (g₁ (r 1), ..., gₙ (r n))` where `r` ranges over all functions `r`. This follows from
multilinearity by expanding successively with respect to each coordinate. -/
theorem map_sum [∀ i, Fintype (α i)] :
    (f fun i => ∑ j, g' i j) = ∑ r : ∀ i, α i, f fun i => g' i (r i) :=
  f.toMultilinearMap.map_sum _


/-- Reinterpret a continuous `A`-alternating map as a continuous `R`-alternating map, if `A` is an
algebra over `R` and their actions on all involved modules agree with the action of `R` on `A`. -/
def restrictScalars (f : M [⋀^ι]→L[A] N) : M [⋀^ι]→L[R] N :=
  { f with toContinuousMultilinearMap := f.1.restrictScalars R }


@[simp]
theorem coe_restrictScalars (f : M [⋀^ι]→L[A] N) : ⇑(f.restrictScalars R) = f :=
  rfl


@[simp]
theorem map_update_sub [DecidableEq ι] (m : ι → M) (i : ι) (x y : M) :
    f (update m i (x - y)) = f (update m i x) - f (update m i y) :=
  f.toMultilinearMap.map_update_sub _ _ _ _


@[deprecated (since := "2024-11-03")] protected alias map_sub := map_update_sub


instance : Neg (M [⋀^ι]→L[R] N) :=
  ⟨fun f => { -f.toAlternatingMap with toContinuousMultilinearMap := -f.1 }⟩


@[simp]
theorem coe_neg : ⇑(-f) = -f :=
  rfl


theorem neg_apply (m : ι → M) : (-f) m = -f m :=
  rfl


instance : Sub (M [⋀^ι]→L[R] N) :=
  ⟨fun f g =>
    { f.toAlternatingMap - g.toAlternatingMap with toContinuousMultilinearMap := f.1 - g.1 }⟩


@[simp] theorem coe_sub : ⇑(f - g) = ⇑f - ⇑g := rfl


theorem sub_apply (m : ι → M) : (f - g) m = f m - g m := rfl


instance : AddCommGroup (M [⋀^ι]→L[R] N) :=
  toContinuousMultilinearMap_injective.addCommGroup _ rfl (fun _ _ => rfl) (fun _ => rfl)
    (fun _ _ => rfl) (fun _ _ => rfl) fun _ _ => rfl


theorem map_piecewise_smul [DecidableEq ι] (c : ι → R) (m : ι → M) (s : Finset ι) :
    f (s.piecewise (fun i => c i • m i) m) = (∏ i ∈ s, c i) • f m :=
  f.toMultilinearMap.map_piecewise_smul _ _ _


/-- Multiplicativity of a continuous alternating map along all coordinates at the same time,
writing `f (fun i ↦ c i • m i)` as `(∏ i, c i) • f m`. -/
theorem map_smul_univ [Fintype ι] (c : ι → R) (m : ι → M) :
    (f fun i => c i • m i) = (∏ i, c i) • f m :=
  f.toMultilinearMap.map_smul_univ _ _


instance [ContinuousAdd N] : DistribMulAction R (M [⋀^ι]→L[A] N) :=
  Function.Injective.distribMulAction toMultilinearAddHom
    toContinuousMultilinearMap_injective fun _ _ => rfl


/-- The space of continuous alternating maps over an algebra over `R` is a module over `R`, for the
pointwise addition and scalar multiplication. -/
instance : Module R (M [⋀^ι]→L[A] N) :=
  Function.Injective.module _ toMultilinearAddHom toContinuousMultilinearMap_injective fun _ _ =>
    rfl


/-- Linear map version of the map `toMultilinearMap` associating to a continuous alternating map
the corresponding multilinear map. -/
@[simps]
def toContinuousMultilinearMapLinear :
    M [⋀^ι]→L[A] N →ₗ[R] ContinuousMultilinearMap A (fun _ : ι => M) N where
  toFun := toContinuousMultilinearMap
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


/-- Linear map version of the map `toAlternatingMap`
associating to a continuous alternating map the corresponding alternating map. -/
@[simps (config := .asFn) apply]
def toAlternatingMapLinear : (M [⋀^ι]→L[A] N) →ₗ[R] (M [⋀^ι]→ₗ[A] N) where
  toFun := toAlternatingMap
                 /-
                   R : Type u_1
                   A : Type u_2
                   M : Type u_3
                   N : Type u_4
                   ι : Type u_5
                   inst✝¹¹ : Semiring R
                   inst✝¹⁰ : Semiring A
                   inst✝⁹ : AddCommMonoid M
                   inst✝⁸ : AddCommMonoid N
                   inst✝⁷ : TopologicalSpace M
                   inst✝⁶ : TopologicalSpace N
                   inst✝⁵ : ContinuousAdd N
                   inst✝⁴ : Module A M
                   inst✝³ : Module A N
                   inst✝² : Module R N
                   inst✝¹ : ContinuousConstSMul R N
                   inst✝ : SMulCommClass A R N
                   ⊢ ∀ (x y : ContinuousAlternatingMap A M N ι), Eq (HAdd.hAdd x y).toAlternating …
                 -/
  map_add' := by simp
                 /-
                   🎉 no goals
                 -/
                  /-
                    R : Type u_1
                    A : Type u_2
                    M : Type u_3
                    N : Type u_4
                    ι : Type u_5
                    inst✝¹¹ : Semiring R
                    inst✝¹⁰ : Semiring A
                    inst✝⁹ : AddCommMonoid M
                    inst✝⁸ : AddCommMonoid N
                    inst✝⁷ : TopologicalSpace M
                    inst✝⁶ : TopologicalSpace N
                    inst✝⁵ : ContinuousAdd N
                    inst✝⁴ : Module A M
                    inst✝³ : Module A N
                    inst✝² : Module R N
                    inst✝¹ : ContinuousConstSMul R N
                    inst✝ : SMulCommClass A R N
                    ⊢ ∀ (m : R) (x : ContinuousAlternatingMap A M N ι), Eq ({ toFun := ContinuousA …
                  -/
  map_smul' := by simp
                  /-
                    🎉 no goals
                  -/


/-- `ContinuousAlternatingMap.pi` as a `LinearEquiv`. -/
@[simps (config := { simpRhs := true })]
def piLinearEquiv {ι' : Type*} {M' : ι' → Type*} [∀ i, AddCommMonoid (M' i)]
    [∀ i, TopologicalSpace (M' i)] [∀ i, ContinuousAdd (M' i)] [∀ i, Module R (M' i)]
    [∀ i, Module A (M' i)] [∀ i, SMulCommClass A R (M' i)] [∀ i, ContinuousConstSMul R (M' i)] :
    (∀ i, M [⋀^ι]→L[A] M' i) ≃ₗ[R] M [⋀^ι]→L[A] ∀ i, M' i :=
  { piEquiv with
    map_add' := fun _ _ => rfl
    map_smul' := fun _ _ => rfl }


/-- Given a continuous `R`-alternating map `f` taking values in `R`, `f.smulRight z` is the
continuous alternating map sending `m` to `f m • z`. -/
@[simps! toContinuousMultilinearMap apply]
def smulRight : M [⋀^ι]→L[R] N :=
  { f.toAlternatingMap.smulRight z with toContinuousMultilinearMap := f.1.smulRight z }


/-- `ContinuousAlternatingMap.compContinuousLinearMap` as a bundled `LinearMap`. -/
@[simps]
def compContinuousLinearMapₗ (f : M →L[R] M') : (M' [⋀^ι]→L[R] N) →ₗ[R] (M [⋀^ι]→L[R] N) where
  toFun g := g.compContinuousLinearMap f
                      /-
                        R : Type u_1
                        M : Type u_2
                        M' : Type u_3
                        N : Type u_4
                        N' : Type u_5
                        ι : Type u_6
                        inst✝¹⁶ : CommSemiring R
                        inst✝¹⁵ : AddCommMonoid M
                        inst✝¹⁴ : Module R M
                        inst✝¹³ : TopologicalSpace M
                        inst✝¹² : AddCommMonoid M'
                        inst✝¹¹ : Module R M'
                        inst✝¹⁰ : TopologicalSpace M'
                        inst✝⁹ : AddCommMonoid N
                        inst✝⁸ : Module R N
                        inst✝⁷ : TopologicalSpace N
                        inst✝⁶ : ContinuousAdd N
                        inst✝⁵ : ContinuousConstSMul R N
                        inst✝⁴ : AddCommMonoid N'
                        inst✝³ : Module R N'
                        inst✝² : TopologicalSpace N'
                        inst✝¹ : ContinuousAdd N'
                        inst✝ : ContinuousConstSMul R N'
                        f : ContinuousLinearMap (RingHom.id R) M M'
                        g g' : ContinuousAlternatingMap R M' N ι
                        ⊢ Eq ((fun g => g.compContinuousLinearMap f) (HAdd.hAdd g g')) (HAdd.hAdd ((fu …
                      -/
  map_add' g g' := by ext; simp
                           /-
                             🎉 no goals
                           -/
                      /-
                        R : Type u_1
                        M : Type u_2
                        M' : Type u_3
                        N : Type u_4
                        N' : Type u_5
                        ι : Type u_6
                        inst✝¹⁶ : CommSemiring R
                        inst✝¹⁵ : AddCommMonoid M
                        inst✝¹⁴ : Module R M
                        inst✝¹³ : TopologicalSpace M
                        inst✝¹² : AddCommMonoid M'
                        inst✝¹¹ : Module R M'
                        inst✝¹⁰ : TopologicalSpace M'
                        inst✝⁹ : AddCommMonoid N
                        inst✝⁸ : Module R N
                        inst✝⁷ : TopologicalSpace N
                        inst✝⁶ : ContinuousAdd N
                        inst✝⁵ : ContinuousConstSMul R N
                        inst✝⁴ : AddCommMonoid N'
                        inst✝³ : Module R N'
                        inst✝² : TopologicalSpace N'
                        inst✝¹ : ContinuousAdd N'
                        inst✝ : ContinuousConstSMul R N'
                        f : ContinuousLinearMap (RingHom.id R) M M'
                        c : R
                        g : ContinuousAlternatingMap R M' N ι
                        ⊢ Eq ({ toFun := fun g => g.compContinuousLinearMap f, map_add' := ⋯ }.toFun ( …
                      -/
  map_smul' c g := by ext; simp
                           /-
                             🎉 no goals
                           -/


/-- `ContinuousLinearMap.compContinuousAlternatingMap` as a bundled bilinear map. -/
def _root_.ContinuousLinearMap.compContinuousAlternatingMapₗ :
    (N →L[R] N') →ₗ[R] (M [⋀^ι]→L[R] N) →ₗ[R] (M [⋀^ι]→L[R] N') :=
  LinearMap.mk₂ R ContinuousLinearMap.compContinuousAlternatingMap (fun _ _ _ => rfl)
                                          /-
                                            R : Type u_1
                                            M : Type u_2
                                            M' : Type u_3
                                            N : Type u_4
                                            N' : Type u_5
                                            ι : Type u_6
                                            inst✝¹⁶ : CommSemiring R
                                            inst✝¹⁵ : AddCommMonoid M
                                            inst✝¹⁴ : Module R M
                                            inst✝¹³ : TopologicalSpace M
                                            inst✝¹² : AddCommMonoid M'
                                            inst✝¹¹ : Module R M'
                                            inst✝¹⁰ : TopologicalSpace M'
                                            inst✝⁹ : AddCommMonoid N
                                            inst✝⁸ : Module R N
                                            inst✝⁷ : TopologicalSpace N
                                            inst✝⁶ : ContinuousAdd N
                                            inst✝⁵ : ContinuousConstSMul R N
                                            inst✝⁴ : AddCommMonoid N'
                                            inst✝³ : Module R N'
                                            inst✝² : TopologicalSpace N'
                                            inst✝¹ : ContinuousAdd N'
                                            inst✝ : ContinuousConstSMul R N'
                                            f : ContinuousLinearMap (RingHom.id R) N N'
                                            g₁ g₂ : ContinuousAlternatingMap R M N ι
                                            ⊢ Eq (f.compContinuousAlternatingMap (HAdd.hAdd g₁ g₂)) (HAdd.hAdd (f.compCont …
                                          -/
                                                /-
                                                  🎉 no goals
                                                -/
    (fun _ _ _ => rfl) (fun f g₁ g₂ => by ext1; apply f.map_add) fun c f g => by ext1; simp
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


/-- Alternatization of a continuous multilinear map. -/
@[simps (config := .lemmasOnly) apply_toContinuousMultilinearMap]
def alternatization : ContinuousMultilinearMap R (fun _ : ι => M) N →+ M [⋀^ι]→L[R] N where
  toFun f :=
    { toContinuousMultilinearMap := ∑ σ : Equiv.Perm ι, Equiv.Perm.sign σ • f.domDomCongr σ
      map_eq_zero_of_eq' := fun v i j hv hne => by
        simpa [MultilinearMap.alternatization_apply]
          using f.1.alternatization.map_eq_zero_of_eq' v i j hv hne }
                  /-
                    R : Type u_1
                    M : Type u_2
                    N : Type u_3
                    ι : Type u_4
                    inst✝⁹ : Semiring R
                    inst✝⁸ : AddCommMonoid M
                    inst✝⁷ : Module R M
                    inst✝⁶ : TopologicalSpace M
                    inst✝⁵ : AddCommGroup N
                    inst✝⁴ : Module R N
                    inst✝³ : TopologicalSpace N
                    inst✝² : TopologicalAddGroup N
                    inst✝¹ : Fintype ι
                    inst✝ : DecidableEq ι
                    f : ContinuousMultilinearMap R (fun x => M) N
                    ⊢ Eq ((fun f => { toContinuousMultilinearMap := Finset.univ.sum fun σ => HSMul …
                  -/
  map_zero' := by ext; simp
                       /-
                         🎉 no goals
                       -/
                     /-
                       R : Type u_1
                       M : Type u_2
                       N : Type u_3
                       ι : Type u_4
                       inst✝⁹ : Semiring R
                       inst✝⁸ : AddCommMonoid M
                       inst✝⁷ : Module R M
                       inst✝⁶ : TopologicalSpace M
                       inst✝⁵ : AddCommGroup N
                       inst✝⁴ : Module R N
                       inst✝³ : TopologicalSpace N
                       inst✝² : TopologicalAddGroup N
                       inst✝¹ : Fintype ι
                       inst✝ : DecidableEq ι
                       f x✝¹ x✝ : ContinuousMultilinearMap R (fun x => M) N
                       ⊢ Eq ({ toFun := fun f => { toContinuousMultilinearMap := Finset.univ.sum fun  …
                     -/
  map_add' _ _ := by ext; simp [Finset.sum_add_distrib]
                          /-
                            🎉 no goals
                          -/


theorem alternatization_apply_apply (v : ι → M) :
    alternatization f v = ∑ σ : Equiv.Perm ι, Equiv.Perm.sign σ • f (v ∘ σ) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    ι : Type u_4
    inst✝⁹ : Semiring R
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : Module R M
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    inst✝³ : TopologicalSpace N
    inst✝² : TopologicalAddGroup N
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    f : ContinuousMultilinearMap R (fun x => M) N
    v : ι → M
    ⊢ Eq ((ContinuousMultilinearMap.alternatization f) v) (Finset.univ.sum fun σ = …
  -/
  simp [alternatization, Function.comp_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem alternatization_apply_toAlternatingMap :
    (alternatization f).toAlternatingMap = MultilinearMap.alternatization f.1 := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    ι : Type u_4
    inst✝⁹ : Semiring R
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : Module R M
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    inst✝³ : TopologicalSpace N
    inst✝² : TopologicalAddGroup N
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    f : ContinuousMultilinearMap R (fun x => M) N
    ⊢ Eq (ContinuousMultilinearMap.alternatization f).toAlternatingMap (Multilinea …
  -/
  ext v
  /-
    case H
    R : Type u_1
    M : Type u_2
    N : Type u_3
    ι : Type u_4
    inst✝⁹ : Semiring R
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : Module R M
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    inst✝³ : TopologicalSpace N
    inst✝² : TopologicalAddGroup N
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    f : ContinuousMultilinearMap R (fun x => M) N
    v : ι → M
    ⊢ Eq ((ContinuousMultilinearMap.alternatization f).toAlternatingMap v) ((Multi …
  -/
  simp [alternatization_apply_apply, MultilinearMap.alternatization_apply, Function.comp_def]
  /-
    🎉 no goals
  -/



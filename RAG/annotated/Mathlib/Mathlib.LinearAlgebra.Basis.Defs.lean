/-- A `Basis ι R M` for a module `M` is the type of `ι`-indexed `R`-bases of `M`.

The basis vectors are available as `DFunLike.coe (b : Basis ι R M) : ι → M`.
To turn a linear independent family of vectors spanning `M` into a basis, use `Basis.mk`.
They are internally represented as linear equivs `M ≃ₗ[R] (ι →₀ R)`,
available as `Basis.repr`.
-/
structure Basis where
  /-- `Basis.ofRepr` constructs a basis given an assignment of coordinates to each vector. -/
  ofRepr ::
    /-- `repr` is the linear equivalence sending a vector `x` to its coordinates:
    the `c`s such that `x = ∑ i, c i`. -/
    repr : M ≃ₗ[R] ι →₀ R


instance uniqueBasis [Subsingleton R] : Unique (Basis ι R M) :=
                              /-
                                ι : Type u_1
                                ι' : Type u_2
                                R : Type u_3
                                R₂ : Type u_4
                                K : Type u_5
                                M : Type u_6
                                M' : Type u_7
                                M'' : Type u_8
                                V : Type u
                                V' : Type u_9
                                inst✝⁵ : Semiring R
                                inst✝⁴ : AddCommMonoid M
                                inst✝³ : Module R M
                                inst✝² : AddCommMonoid M'
                                inst✝¹ : Module R M'
                                inst✝ : Subsingleton R
                                x✝ : Basis ι R M
                                b : LinearEquiv (RingHom.id R) M (Finsupp ι R)
                                ⊢ Eq { repr := b } Inhabited.default
                              -/
  ⟨⟨⟨default⟩⟩, fun ⟨b⟩ => by rw [Subsingleton.elim b]⟩
                              /-
                                🎉 no goals
                              -/


instance : Inhabited (Basis ι R (ι →₀ R)) :=
  ⟨.ofRepr (LinearEquiv.refl _ _)⟩


theorem repr_injective : Injective (repr : Basis ι R M → M ≃ₗ[R] ι →₀ R) := fun f g h => by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f g : Basis ι R M
    h : Eq f.repr g.repr
    ⊢ Eq f g
  -/
  cases f; cases g; congr
                    /-
                      🎉 no goals
                    -/


/-- `b i` is the `i`th basis vector. -/
instance instFunLike : FunLike (Basis ι R M) ι M where
  coe b i := b.repr.symm (Finsupp.single i 1)
  coe_injective' f g h := repr_injective <| LinearEquiv.symm_bijective.injective <|
                                            /-
                                              ι : Type u_1
                                              ι' : Type u_2
                                              R : Type u_3
                                              R₂ : Type u_4
                                              K : Type u_5
                                              M : Type u_6
                                              M' : Type u_7
                                              M'' : Type u_8
                                              V : Type u
                                              V' : Type u_9
                                              inst✝⁴ : Semiring R
                                              inst✝³ : AddCommMonoid M
                                              inst✝² : Module R M
                                              inst✝¹ : AddCommMonoid M'
                                              inst✝ : Module R M'
                                              b b₁ : Basis ι R M
                                              i : ι
                                              c : R
                                              x : M
                                              f g : Basis ι R M
                                              h : Eq ((fun b i => b.repr.symm (Finsupp.single i 1)) f) ((fun b i => b.repr.s …
                                              ⊢ Eq ↑f.repr.symm ↑g.repr.symm
                                            -/
    LinearEquiv.toLinearMap_injective <| by ext; exact congr_fun h _
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem coe_ofRepr (e : M ≃ₗ[R] ι →₀ R) : ⇑(ofRepr e) = fun i => e.symm (Finsupp.single i 1) :=
  rfl


protected theorem injective [Nontrivial R] : Injective b :=
  b.repr.symm.injective.comp fun _ _ => (Finsupp.single_left_inj (one_ne_zero : (1 : R) ≠ 0)).mp


theorem repr_symm_single_one : b.repr.symm (Finsupp.single i 1) = b i :=
  rfl


theorem repr_symm_single : b.repr.symm (Finsupp.single i c) = c • b i :=
  calc
    b.repr.symm (Finsupp.single i c) = b.repr.symm (c • Finsupp.single i (1 : R)) := by
      /-
        ι : Type u_1
        R : Type u_3
        M : Type u_6
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        b : Basis ι R M
        i : ι
        c : R
        ⊢ Eq (b.repr.symm (Finsupp.single i c)) (b.repr.symm (HSMul.hSMul c (Finsupp.s …
      -/
      { rw [Finsupp.smul_single', mul_one] }
      /-
        🎉 no goals
      -/
                      /-
                        ι : Type u_1
                        R : Type u_3
                        M : Type u_6
                        inst✝² : Semiring R
                        inst✝¹ : AddCommMonoid M
                        inst✝ : Module R M
                        b : Basis ι R M
                        i : ι
                        c : R
                        ⊢ Eq (b.repr.symm (HSMul.hSMul c (Finsupp.single i 1))) (HSMul.hSMul c (b i))
                      -/
    _ = c • b i := by rw [LinearEquiv.map_smul, repr_symm_single_one]
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem repr_self : b.repr (b i) = Finsupp.single i 1 :=
  LinearEquiv.apply_symm_apply _ _


theorem repr_self_apply (j) [Decidable (i = j)] : b.repr (b i) j = if i = j then 1 else 0 := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    b : Basis ι R M
    i j : ι
    inst✝ : Decidable (Eq i j)
    ⊢ Eq ((b.repr (b i)) j) (ite (Eq i j) 1 0)
  -/
  rw [repr_self, Finsupp.single_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem repr_symm_apply (v) : b.repr.symm v = Finsupp.linearCombination R b v :=
  calc
                                                             /-
                                                               ι : Type u_1
                                                               R : Type u_3
                                                               M : Type u_6
                                                               inst✝² : Semiring R
                                                               inst✝¹ : AddCommMonoid M
                                                               inst✝ : Module R M
                                                               b : Basis ι R M
                                                               v : Finsupp ι R
                                                               ⊢ Eq (b.repr.symm v) (b.repr.symm (v.sum Finsupp.single))
                                                             -/
    b.repr.symm v = b.repr.symm (v.sum Finsupp.single) := by simp
                                                             /-
                                                               🎉 no goals
                                                             -/
    _ = v.sum fun i vi => b.repr.symm (Finsupp.single i vi) := map_finsupp_sum ..
    _ = Finsupp.linearCombination R b v := by simp only [repr_symm_single,
                                                         Finsupp.linearCombination_apply]


@[simp]
theorem coe_repr_symm : ↑b.repr.symm = Finsupp.linearCombination R b :=
  LinearMap.ext fun v => b.repr_symm_apply v


@[simp]
theorem repr_linearCombination (v) : b.repr (Finsupp.linearCombination _ b v) = v := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    v : Finsupp ι R
    ⊢ Eq (b.repr ((Finsupp.linearCombination R ⇑b) v)) v
  -/
  rw [← b.coe_repr_symm]
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    v : Finsupp ι R
    ⊢ Eq (b.repr (↑b.repr.symm v)) v
  -/
  exact b.repr.apply_symm_apply v
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias repr_total := repr_linearCombination


@[simp]
theorem linearCombination_repr : Finsupp.linearCombination _ b (b.repr x) = x := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    x : M
    ⊢ Eq ((Finsupp.linearCombination R ⇑b) (b.repr x)) x
  -/
  rw [← b.coe_repr_symm]
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    x : M
    ⊢ Eq (↑b.repr.symm (b.repr x)) x
  -/
  exact b.repr.symm_apply_apply x
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-08-29")] alias total_repr := linearCombination_repr


theorem repr_range : LinearMap.range (b.repr : M →ₗ[R] ι →₀ R) = Finsupp.supported R R univ := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    ⊢ Eq (LinearMap.range ↑b.repr) (Finsupp.supported R R Set.univ)
  -/
  rw [LinearEquiv.range, Finsupp.supported_univ]
  /-
    🎉 no goals
  -/


theorem mem_span_repr_support (m : M) : m ∈ span R (b '' (b.repr m).support) :=
  (Finsupp.mem_span_image_iff_linearCombination _).2
                  /-
                    ι : Type u_1
                    R : Type u_3
                    M : Type u_6
                    inst✝² : Semiring R
                    inst✝¹ : AddCommMonoid M
                    inst✝ : Module R M
                    b : Basis ι R M
                    m : M
                    ⊢ And (Membership.mem (Finsupp.supported R R ↑(b.repr m).support) (b.repr m))  …
                  -/
    ⟨b.repr m, by simp [Finsupp.mem_supported_support]⟩
                  /-
                    🎉 no goals
                  -/


theorem repr_support_subset_of_mem_span (s : Set ι) {m : M}
    (hm : m ∈ span R (b '' s)) : ↑(b.repr m).support ⊆ s := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    s : Set ι
    m : M
    hm : Membership.mem (Submodule.span R (Set.image (⇑b) s)) m
    ⊢ HasSubset.Subset (↑(b.repr m).support) s
  -/
  rcases (Finsupp.mem_span_image_iff_linearCombination _).1 hm with ⟨l, hl, rfl⟩
  /-
    case intro.intro
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    s : Set ι
    l : Finsupp ι R
    hl : Membership.mem (Finsupp.supported R R s) l
    hm : Membership.mem (Submodule.span R (Set.image (⇑b) s)) ((Finsupp.linearComb …
    ⊢ HasSubset.Subset (↑(b.repr ((Finsupp.linearCombination R ⇑b) l)).support) s
  -/
  rwa [repr_linearCombination, ← Finsupp.mem_supported R l]
  /-
    🎉 no goals
  -/


theorem mem_span_image {m : M} {s : Set ι} : m ∈ span R (b '' s) ↔ ↑(b.repr m).support ⊆ s :=
  ⟨repr_support_subset_of_mem_span _ _, fun h ↦
    span_mono (image_subset _ h) (mem_span_repr_support b _)⟩


@[simp]
theorem self_mem_span_image [Nontrivial R] {i : ι} {s : Set ι} :
    b i ∈ span R (b '' s) ↔ i ∈ s := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    b : Basis ι R M
    inst✝ : Nontrivial R
    i : ι
    s : Set ι
    ⊢ Iff (Membership.mem (Submodule.span R (Set.image (⇑b) s)) (b i)) (Membership …
  -/
  simp [mem_span_image, Finsupp.support_single_ne_zero]
  /-
    🎉 no goals
  -/


/-- `b.coord i` is the linear function giving the `i`'th coordinate of a vector
with respect to the basis `b`.

`b.coord i` is an element of the dual space. In particular, for
finite-dimensional spaces it is the `ι`th basis vector of the dual space.
-/
@[simps!]
def coord : M →ₗ[R] R :=
  Finsupp.lapply i ∘ₗ ↑b.repr


theorem forall_coord_eq_zero_iff {x : M} : (∀ i, b.coord i x = 0) ↔ x = 0 :=
                /-
                  ι : Type u_1
                  R : Type u_3
                  M : Type u_6
                  inst✝² : Semiring R
                  inst✝¹ : AddCommMonoid M
                  inst✝ : Module R M
                  b : Basis ι R M
                  x : M
                  ⊢ Iff (∀ (i : ι), Eq ((b.coord i) x) 0) (Eq (b.repr x) 0)
                -/
  Iff.trans (by simp only [b.coord_apply, DFunLike.ext_iff, Finsupp.zero_apply])
                /-
                  🎉 no goals
                -/
    b.repr.map_eq_zero_iff


/-- The sum of the coordinates of an element `m : M` with respect to a basis. -/
noncomputable def sumCoords : M →ₗ[R] R :=
  (Finsupp.lsum ℕ fun _ => LinearMap.id) ∘ₗ (b.repr : M →ₗ[R] ι →₀ R)


@[simp]
theorem coe_sumCoords : (b.sumCoords : M → R) = fun m => (b.repr m).sum fun _ => id :=
  rfl


@[simp high]
theorem coe_sumCoords_of_fintype [Fintype ι] : (b.sumCoords : M → R) = ∑ i, b.coord i := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    b : Basis ι R M
    inst✝ : Fintype ι
    ⊢ Eq ⇑b.sumCoords ⇑(Finset.univ.sum fun i => b.coord i)
  -/
  ext m
  -- Porting note: - `eq_self_iff_true`
  --               + `comp_apply` `LinearMap.coeFn_sum`
  simp only [sumCoords, Finsupp.sum_fintype, LinearMap.id_coe, LinearEquiv.coe_coe, coord_apply,
    id, Fintype.sum_apply, imp_true_iff, Finsupp.coe_lsum, LinearMap.coe_comp, comp_apply,
    LinearMap.coeFn_sum]


@[simp]
theorem sumCoords_self_apply : b.sumCoords (b i) = 1 := by
  simp only [Basis.sumCoords, LinearMap.id_coe, LinearEquiv.coe_coe, id, Basis.repr_self,
    Function.comp_apply, Finsupp.coe_lsum, LinearMap.coe_comp, Finsupp.sum_single_index]


theorem dvd_coord_smul (i : ι) (m : M) (r : R) : r ∣ b.coord i (r • m) :=
                   /-
                     ι : Type u_1
                     R : Type u_3
                     M : Type u_6
                     inst✝² : Semiring R
                     inst✝¹ : AddCommMonoid M
                     inst✝ : Module R M
                     b : Basis ι R M
                     i : ι
                     m : M
                     r : R
                     ⊢ Eq ((b.coord i) (HSMul.hSMul r m)) (HMul.hMul r ((b.coord i) m))
                   -/
  ⟨b.coord i m, by simp⟩
                   /-
                     🎉 no goals
                   -/


theorem coord_repr_symm (b : Basis ι R M) (i : ι) (f : ι →₀ R) :
    b.coord i (b.repr.symm f) = f i := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    i : ι
    f : Finsupp ι R
    ⊢ Eq ((b.coord i) (b.repr.symm f)) (f i)
  -/
  simp only [repr_symm_apply, coord_apply, repr_linearCombination]
  /-
    🎉 no goals
  -/


/-- Two linear maps are equal if they are equal on basis vectors. -/
theorem ext {f₁ f₂ : M →ₛₗ[σ] M₁} (h : ∀ i, f₁ (b i) = f₂ (b i)) : f₁ = f₂ := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝⁵ : Semiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    b : Basis ι R M
    R₁ : Type u_10
    inst✝² : Semiring R₁
    σ : RingHom R R₁
    M₁ : Type u_11
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    f₁ f₂ : LinearMap σ M M₁
    h : ∀ (i : ι), Eq (f₁ (b i)) (f₂ (b i))
    ⊢ Eq f₁ f₂
  -/
  ext x
  /-
    case h
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝⁵ : Semiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    b : Basis ι R M
    R₁ : Type u_10
    inst✝² : Semiring R₁
    σ : RingHom R R₁
    M₁ : Type u_11
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    f₁ f₂ : LinearMap σ M M₁
    h : ∀ (i : ι), Eq (f₁ (b i)) (f₂ (b i))
    x : M
    ⊢ Eq (f₁ x) (f₂ x)
  -/
  rw [← b.linearCombination_repr x, Finsupp.linearCombination_apply, Finsupp.sum]
  /-
    case h
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝⁵ : Semiring R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    b : Basis ι R M
    R₁ : Type u_10
    inst✝² : Semiring R₁
    σ : RingHom R R₁
    M₁ : Type u_11
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    f₁ f₂ : LinearMap σ M M₁
    h : ∀ (i : ι), Eq (f₁ (b i)) (f₂ (b i))
    x : M
    ⊢ Eq (f₁ ((b.repr x).support.sum fun a => HSMul.hSMul ((b.repr x) a) (b a))) ( …
  -/
  simp only [map_sum, LinearMap.map_smulₛₗ, h]
  /-
    🎉 no goals
  -/


/-- Two linear equivs are equal if they are equal on basis vectors. -/
theorem ext' {f₁ f₂ : M ≃ₛₗ[σ] M₁} (h : ∀ i, f₁ (b i) = f₂ (b i)) : f₁ = f₂ := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝⁷ : Semiring R
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    b : Basis ι R M
    R₁ : Type u_10
    inst✝⁴ : Semiring R₁
    σ : RingHom R R₁
    σ' : RingHom R₁ R
    inst✝³ : RingHomInvPair σ σ'
    inst✝² : RingHomInvPair σ' σ
    M₁ : Type u_11
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    f₁ f₂ : LinearEquiv σ M M₁
    h : ∀ (i : ι), Eq (f₁ (b i)) (f₂ (b i))
    ⊢ Eq f₁ f₂
  -/
  ext x
  /-
    case h
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝⁷ : Semiring R
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    b : Basis ι R M
    R₁ : Type u_10
    inst✝⁴ : Semiring R₁
    σ : RingHom R R₁
    σ' : RingHom R₁ R
    inst✝³ : RingHomInvPair σ σ'
    inst✝² : RingHomInvPair σ' σ
    M₁ : Type u_11
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    f₁ f₂ : LinearEquiv σ M M₁
    h : ∀ (i : ι), Eq (f₁ (b i)) (f₂ (b i))
    x : M
    ⊢ Eq (f₁ x) (f₂ x)
  -/
  rw [← b.linearCombination_repr x, Finsupp.linearCombination_apply, Finsupp.sum]
  /-
    case h
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝⁷ : Semiring R
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    b : Basis ι R M
    R₁ : Type u_10
    inst✝⁴ : Semiring R₁
    σ : RingHom R R₁
    σ' : RingHom R₁ R
    inst✝³ : RingHomInvPair σ σ'
    inst✝² : RingHomInvPair σ' σ
    M₁ : Type u_11
    inst✝¹ : AddCommMonoid M₁
    inst✝ : Module R₁ M₁
    f₁ f₂ : LinearEquiv σ M M₁
    h : ∀ (i : ι), Eq (f₁ (b i)) (f₂ (b i))
    x : M
    ⊢ Eq (f₁ ((b.repr x).support.sum fun a => HSMul.hSMul ((b.repr x) a) (b a))) ( …
  -/
  simp only [map_sum, LinearEquiv.map_smulₛₗ, h]
  /-
    🎉 no goals
  -/


/-- Two elements are equal iff their coordinates are equal. -/
theorem ext_elem_iff {x y : M} : x = y ↔ ∀ i, b.repr x i = b.repr y i := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    x y : M
    ⊢ Iff (Eq x y) (∀ (i : ι), Eq ((b.repr x) i) ((b.repr y) i))
  -/
  simp only [← DFunLike.ext_iff, EmbeddingLike.apply_eq_iff_eq]
  /-
    🎉 no goals
  -/


alias ⟨_, _root_.Basis.ext_elem⟩ := ext_elem_iff


theorem repr_eq_iff {b : Basis ι R M} {f : M →ₗ[R] ι →₀ R} :
    ↑b.repr = f ↔ ∀ i, f (b i) = Finsupp.single i 1 :=
  ⟨fun h i => h ▸ b.repr_self i, fun h => b.ext fun i => (b.repr_self i).trans (h i).symm⟩


theorem repr_eq_iff' {b : Basis ι R M} {f : M ≃ₗ[R] ι →₀ R} :
    b.repr = f ↔ ∀ i, f (b i) = Finsupp.single i 1 :=
  ⟨fun h i => h ▸ b.repr_self i, fun h => b.ext' fun i => (b.repr_self i).trans (h i).symm⟩


theorem apply_eq_iff {b : Basis ι R M} {x : M} {i : ι} : b i = x ↔ b.repr x = Finsupp.single i 1 :=
  ⟨fun h => h ▸ b.repr_self i, fun h => b.repr.injective ((b.repr_self i).trans h.symm)⟩


/-- An unbundled version of `repr_eq_iff` -/
theorem repr_apply_eq (f : M → ι → R) (hadd : ∀ x y, f (x + y) = f x + f y)
    (hsmul : ∀ (c : R) (x : M), f (c • x) = c • f x) (f_eq : ∀ i, f (b i) = Finsupp.single i 1)
    (x : M) (i : ι) : b.repr x i = f x i := by
  let f_i : M →ₗ[R] R :=
    { toFun := fun x => f x i
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
      map_add' := fun _ _ => by beta_reduce; rw [hadd, Pi.add_apply]
      map_smul' := fun _ _ => by simp [hsmul, Pi.smul_apply] }
  have : Finsupp.lapply i ∘ₗ ↑b.repr = f_i := by
    refine b.ext fun j => ?_
    show b.repr (b j) i = f (b j) i
    rw [b.repr_self, f_eq]
  calc
    b.repr x i = f_i x := by
      { rw [← this]
        rfl }
    _ = f x i := rfl


/-- Two bases are equal if they assign the same coordinates. -/
theorem eq_ofRepr_eq_repr {b₁ b₂ : Basis ι R M} (h : ∀ x i, b₁.repr x i = b₂.repr x i) : b₁ = b₂ :=
                       /-
                         ι : Type u_1
                         R : Type u_3
                         M : Type u_6
                         inst✝² : Semiring R
                         inst✝¹ : AddCommMonoid M
                         inst✝ : Module R M
                         b₁ b₂ : Basis ι R M
                         h : ∀ (x : M) (i : ι), Eq ((b₁.repr x) i) ((b₂.repr x) i)
                         ⊢ Eq b₁.repr b₂.repr
                       -/
  repr_injective <| by ext; apply h
                            /-
                              🎉 no goals
                            -/


/-- Two bases are equal if their basis vectors are the same. -/
@[ext]
theorem eq_of_apply_eq {b₁ b₂ : Basis ι R M} : (∀ i, b₁ i = b₂ i) → b₁ = b₂ :=
  DFunLike.ext _ _


/-- Apply the linear equivalence `f` to the basis vectors. -/
@[simps]
protected def map : Basis ι R M' :=
  ofRepr (f.symm.trans b.repr)


@[simp]
theorem map_apply (i) : b.map f i = f (b i) :=
  rfl


theorem coe_map : (b.map f : ι → M') = f ∘ b :=
  rfl


/-- The action on a `Basis` by acting on each element.

See also `Basis.unitsSMul` and `Basis.groupSMul`, for the cases when a different action is applied
to each basis element. -/
instance : SMul G (Basis ι R M) where
  smul g b := b.map <| DistribMulAction.toLinearEquiv _ _ g


@[simp]
theorem smul_apply (g : G) (b : Basis ι R M) (i : ι) : (g • b) i = g • b i := rfl


@[norm_cast] theorem coe_smul (g : G) (b : Basis ι R M) : ⇑(g • b) = g • ⇑b := rfl


/-- When the group in question is the automorphisms, `•` coincides with `Basis.map`. -/
@[simp]
theorem smul_eq_map (g : M ≃ₗ[R] M) (b : Basis ι R M) : g • b = b.map g := rfl


@[simp] theorem repr_smul (g : G) (b : Basis ι R M) :
    (g • b).repr = (DistribMulAction.toLinearEquiv _ _ g).symm.trans b.repr := rfl


instance : MulAction G (Basis ι R M) :=
  Function.Injective.mulAction _ DFunLike.coe_injective coe_smul


instance [SMulCommClass G G' M] : SMulCommClass G G' (Basis ι R M) where
  smul_comm _g _g' _b := DFunLike.ext _ _ fun _ => smul_comm _ _ _


instance [SMul G G'] [IsScalarTower G G' M] : IsScalarTower G G' (Basis ι R M) where
  smul_assoc _g _g' _b := DFunLike.ext _ _ fun _ => smul_assoc _ _ _


/-- If `R` and `R'` are isomorphic rings that act identically on a module `M`,
then a basis for `M` as `R`-module is also a basis for `M` as `R'`-module.

See also `Basis.algebraMapCoeffs` for the case where `f` is equal to `algebraMap`.
-/
@[simps (config := { simpRhs := true })]
def mapCoeffs (h : ∀ (c) (x : M), f c • x = c • x) : Basis ι R' M := by
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    R₂ : Type u_4
    K : Type u_5
    M : Type u_6
    M' : Type u_7
    M'' : Type u_8
    V : Type u
    V' : Type u_9
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid M'
    inst✝² : Module R M'
    b b₁ : Basis ι R M
    i : ι
    c : R
    x : M
    R' : Type u_10
    inst✝¹ : Semiring R'
    inst✝ : Module R' M
    f : RingEquiv R R'
    h : ∀ (c : R) (x : M), Eq (HSMul.hSMul (f c) x) (HSMul.hSMul c x)
    ⊢ Basis ι R' M
  -/
  letI : Module R' R := Module.compHom R (↑f.symm : R' →+* R)
  haveI : IsScalarTower R' R M :=
    { smul_assoc := fun x y z => by
        -- Porting note: `dsimp [(· • ·)]` is unavailable because
        --               `HSMul.hsmul` becomes `SMul.smul`.
        change (f.symm x * y) • z = x • (y • z)
        rw [mul_smul, ← h, f.apply_symm_apply] }
  exact ofRepr <| (b.repr.restrictScalars R').trans <|
    Finsupp.mapRange.linearEquiv (Module.compHom.toLinearEquiv f.symm).symm


theorem mapCoeffs_apply (i : ι) : b.mapCoeffs f h i = b i :=
  apply_eq_iff.mpr <| by
    -- Porting note: in Lean 3, these were automatically inferred from the definition of
    -- `mapCoeffs`.
    /-
      ι : Type u_1
      R : Type u_3
      M : Type u_6
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      b : Basis ι R M
      R' : Type u_10
      inst✝¹ : Semiring R'
      inst✝ : Module R' M
      f : RingEquiv R R'
      h : ∀ (c : R) (x : M), Eq (HSMul.hSMul (f c) x) (HSMul.hSMul c x)
      i : ι
      ⊢ Eq ((b.mapCoeffs f h).repr (b i)) (Finsupp.single i 1)
    -/
    letI : Module R' R := Module.compHom R (↑f.symm : R' →+* R)
    haveI : IsScalarTower R' R M :=
    { smul_assoc := fun x y z => by
        -- Porting note: `dsimp [(· • ·)]` is unavailable because
        --               `HSMul.hsmul` becomes `SMul.smul`.
        change (f.symm x * y) • z = x • (y • z)
        rw [mul_smul, ← h, f.apply_symm_apply] }
    /-
      ι : Type u_1
      R : Type u_3
      M : Type u_6
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      b : Basis ι R M
      R' : Type u_10
      inst✝¹ : Semiring R'
      inst✝ : Module R' M
      f : RingEquiv R R'
      h : ∀ (c : R) (x : M), Eq (HSMul.hSMul (f c) x) (HSMul.hSMul c x)
      i : ι
      this✝ : Module R' R := Module.compHom R ↑f.symm
      this : IsScalarTower R' R M
      ⊢ Eq ((b.mapCoeffs f h).repr (b i)) (Finsupp.single i 1)
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_mapCoeffs : (b.mapCoeffs f h : ι → M) = b :=
  funext <| b.mapCoeffs_apply f h


/-- `b.reindex (e : ι ≃ ι')` is a basis indexed by `ι'` -/
def reindex : Basis ι' R M :=
  .ofRepr (b.repr.trans (Finsupp.domLCongr e))


theorem reindex_apply (i' : ι') : b.reindex e i' = b (e.symm i') :=
  show (b.repr.trans (Finsupp.domLCongr e)).symm (Finsupp.single i' 1) =
    b.repr.symm (Finsupp.single (e.symm i') 1)
     /-
       ι : Type u_1
       ι' : Type u_2
       R : Type u_3
       M : Type u_6
       inst✝² : Semiring R
       inst✝¹ : AddCommMonoid M
       inst✝ : Module R M
       b : Basis ι R M
       e : Equiv ι ι'
       i' : ι'
       ⊢ Eq ((b.repr.trans (Finsupp.domLCongr e)).symm (Finsupp.single i' 1)) (b.repr …
     -/
  by rw [LinearEquiv.symm_trans_apply, Finsupp.domLCongr_symm, Finsupp.domLCongr_single]
     /-
       🎉 no goals
     -/


@[simp]
theorem coe_reindex : (b.reindex e : ι' → M) = b ∘ e.symm :=
  funext (b.reindex_apply e)


theorem repr_reindex_apply (i' : ι') : (b.reindex e).repr x i' = b.repr x (e.symm i') :=
                                                              /-
                                                                ι : Type u_1
                                                                ι' : Type u_2
                                                                R : Type u_3
                                                                M : Type u_6
                                                                inst✝² : Semiring R
                                                                inst✝¹ : AddCommMonoid M
                                                                inst✝ : Module R M
                                                                b : Basis ι R M
                                                                x : M
                                                                e : Equiv ι ι'
                                                                i' : ι'
                                                                ⊢ Eq (((Finsupp.domLCongr e) (b.repr x)) i') ((b.repr x) (e.symm i'))
                                                              -/
  show (Finsupp.domLCongr e : _ ≃ₗ[R] _) (b.repr x) i' = _ by simp
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem repr_reindex : (b.reindex e).repr x = (b.repr x).mapDomain e :=
                         /-
                           ι : Type u_1
                           ι' : Type u_2
                           R : Type u_3
                           M : Type u_6
                           inst✝² : Semiring R
                           inst✝¹ : AddCommMonoid M
                           inst✝ : Module R M
                           b : Basis ι R M
                           x : M
                           e : Equiv ι ι'
                           ⊢ ∀ (x_1 : ι'), Eq (((b.reindex e).repr x) x_1) ((Finsupp.mapDomain (⇑e) (b.re …
                         -/
  DFunLike.ext _ _ <| by simp [repr_reindex_apply]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem reindex_refl : b.reindex (Equiv.refl ι) = b :=
                             /-
                               ι : Type u_1
                               R : Type u_3
                               M : Type u_6
                               inst✝² : Semiring R
                               inst✝¹ : AddCommMonoid M
                               inst✝ : Module R M
                               b : Basis ι R M
                               i : ι
                               ⊢ Eq ((b.reindex (Equiv.refl ι)) i) (b i)
                             -/
  eq_of_apply_eq fun i => by simp
                             /-
                               🎉 no goals
                             -/


/-- `simp` can prove this as `Basis.coe_reindex` + `EquivLike.range_comp` -/
theorem range_reindex : Set.range (b.reindex e) = Set.range b := by
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    e : Equiv ι ι'
    ⊢ Eq (Set.range ⇑(b.reindex e)) (Set.range ⇑b)
  -/
  simp [coe_reindex, range_comp]
  /-
    🎉 no goals
  -/


@[simp]
theorem sumCoords_reindex : (b.reindex e).sumCoords = b.sumCoords := by
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    e : Equiv ι ι'
    ⊢ Eq (b.reindex e).sumCoords b.sumCoords
  -/
  ext x
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    e : Equiv ι ι'
    x : M
    ⊢ Eq ((b.reindex e).sumCoords x) (b.sumCoords x)
  -/
  simp only [coe_sumCoords, repr_reindex]
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    e : Equiv ι ι'
    x : M
    ⊢ Eq ((Finsupp.mapDomain (⇑e) (b.repr x)).sum fun x => id) ((b.repr x).sum fun …
  -/
  exact Finsupp.sum_mapDomain_index (fun _ => rfl) fun _ _ _ => rfl
  /-
    🎉 no goals
  -/


/-- `b.reindex_range` is a basis indexed by `range b`, the basis vectors themselves. -/
def reindexRange : Basis (range b) R M :=
  haveI := Classical.dec (Nontrivial R)
  if h : Nontrivial R then
    letI := h
    b.reindex (Equiv.ofInjective b (Basis.injective b))
  else
    letI : Subsingleton R := not_nontrivial_iff_subsingleton.mp h
    .ofRepr (Module.subsingletonEquiv R M (range b))


theorem reindexRange_self (i : ι) (h := Set.mem_range_self i) : b.reindexRange ⟨b i, h⟩ = b i := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    i : ι
    h : optParam (Membership.mem (Set.range ⇑b) (b i)) ⋯
    ⊢ Eq (b.reindexRange ⟨b i, h⟩) (b i)
  -/
  by_cases htr : Nontrivial R
    /-
      case pos
      ι : Type u_1
      R : Type u_3
      M : Type u_6
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      b : Basis ι R M
      i : ι
      h : optParam (Membership.mem (Set.range ⇑b) (b i)) ⋯
      htr : Nontrivial R
      ⊢ Eq (b.reindexRange ⟨b i, h⟩) (b i)
    -/
  · letI := htr
    simp [htr, reindexRange, reindex_apply, Equiv.apply_ofInjective_symm b.injective,
      Subtype.coe_mk]
    /-
      case neg
      ι : Type u_1
      R : Type u_3
      M : Type u_6
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      b : Basis ι R M
      i : ι
      h : optParam (Membership.mem (Set.range ⇑b) (b i)) ⋯
      htr : Not (Nontrivial R)
      ⊢ Eq (b.reindexRange ⟨b i, h⟩) (b i)
    -/
  · letI : Subsingleton R := not_nontrivial_iff_subsingleton.mp htr
    /-
      case neg
      ι : Type u_1
      R : Type u_3
      M : Type u_6
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      b : Basis ι R M
      i : ι
      h : optParam (Membership.mem (Set.range ⇑b) (b i)) ⋯
      htr : Not (Nontrivial R)
      this : Subsingleton R := not_nontrivial_iff_subsingleton.mp htr
      ⊢ Eq (b.reindexRange ⟨b i, h⟩) (b i)
    -/
    letI := Module.subsingleton R M
    /-
      case neg
      ι : Type u_1
      R : Type u_3
      M : Type u_6
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      b : Basis ι R M
      i : ι
      h : optParam (Membership.mem (Set.range ⇑b) (b i)) ⋯
      htr : Not (Nontrivial R)
      this✝ : Subsingleton R := not_nontrivial_iff_subsingleton.mp htr
      this : Subsingleton M := Module.subsingleton R M
      ⊢ Eq (b.reindexRange ⟨b i, h⟩) (b i)
    -/
    simp [reindexRange, eq_iff_true_of_subsingleton]
    /-
      🎉 no goals
    -/


theorem reindexRange_repr_self (i : ι) :
    b.reindexRange.repr (b i) = Finsupp.single ⟨b i, mem_range_self i⟩ 1 :=
  calc
    b.reindexRange.repr (b i) = b.reindexRange.repr (b.reindexRange ⟨b i, mem_range_self i⟩) :=
      congr_arg _ (b.reindexRange_self _ _).symm
    _ = Finsupp.single ⟨b i, mem_range_self i⟩ 1 := b.reindexRange.repr_self _


@[simp]
theorem reindexRange_apply (x : range b) : b.reindexRange x = x := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    x : ↑(Set.range ⇑b)
    ⊢ Eq (b.reindexRange x) ↑x
  -/
  rcases x with ⟨bi, ⟨i, rfl⟩⟩
  /-
    case mk.intro
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    i : ι
    ⊢ Eq (b.reindexRange ⟨b i, ⋯⟩) ↑⟨b i, ⋯⟩
  -/
  exact b.reindexRange_self i
  /-
    🎉 no goals
  -/


theorem reindexRange_repr' (x : M) {bi : M} {i : ι} (h : b i = bi) :
    b.reindexRange.repr x ⟨bi, ⟨i, h⟩⟩ = b.repr x i := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    x bi : M
    i : ι
    h : Eq (b i) bi
    ⊢ Eq ((b.reindexRange.repr x) ⟨bi, ⋯⟩) ((b.repr x) i)
  -/
  nontriviality
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    x bi : M
    i : ι
    h : Eq (b i) bi
    a✝ : Nontrivial R
    ⊢ Eq ((b.reindexRange.repr x) ⟨bi, ⋯⟩) ((b.repr x) i)
  -/
  subst h
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    x : M
    i : ι
    a✝ : Nontrivial R
    ⊢ Eq ((b.reindexRange.repr x) ⟨b i, ⋯⟩) ((b.repr x) i)
  -/
  apply (b.repr_apply_eq (fun x i => b.reindexRange.repr x ⟨b i, _⟩) _ _ _ x i).symm
    /-
      ι : Type u_1
      R : Type u_3
      M : Type u_6
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      b : Basis ι R M
      x : M
      i : ι
      a✝ : Nontrivial R
      ⊢ ∀ (x y : M), Eq ((fun x i => (b.reindexRange.repr x) ⟨b i, ⋯⟩) (HAdd.hAdd x  …
    -/
  · intro x y
    /-
      ι : Type u_1
      R : Type u_3
      M : Type u_6
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      b : Basis ι R M
      x✝ : M
      i : ι
      a✝ : Nontrivial R
      x y : M
      ⊢ Eq ((fun x i => (b.reindexRange.repr x) ⟨b i, ⋯⟩) (HAdd.hAdd x y)) (HAdd.hAd …
    -/
    ext i
    /-
      case h
      ι : Type u_1
      R : Type u_3
      M : Type u_6
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      b : Basis ι R M
      x✝ : M
      i✝ : ι
      a✝ : Nontrivial R
      x y : M
      i : ι
      ⊢ Eq ((fun x i => (b.reindexRange.repr x) ⟨b i, ⋯⟩) (HAdd.hAdd x y) i) (HAdd.h …
    -/
    simp only [Pi.add_apply, LinearEquiv.map_add, Finsupp.coe_add]
    /-
      🎉 no goals
    -/
    /-
      ι : Type u_1
      R : Type u_3
      M : Type u_6
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      b : Basis ι R M
      x : M
      i : ι
      a✝ : Nontrivial R
      ⊢ ∀ (c : R) (x : M), Eq ((fun x i => (b.reindexRange.repr x) ⟨b i, ⋯⟩) (HSMul. …
    -/
  · intro c x
    /-
      ι : Type u_1
      R : Type u_3
      M : Type u_6
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      b : Basis ι R M
      x✝ : M
      i : ι
      a✝ : Nontrivial R
      c : R
      x : M
      ⊢ Eq ((fun x i => (b.reindexRange.repr x) ⟨b i, ⋯⟩) (HSMul.hSMul c x)) (HSMul. …
    -/
    ext i
    /-
      case h
      ι : Type u_1
      R : Type u_3
      M : Type u_6
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      b : Basis ι R M
      x✝ : M
      i✝ : ι
      a✝ : Nontrivial R
      c : R
      x : M
      i : ι
      ⊢ Eq ((fun x i => (b.reindexRange.repr x) ⟨b i, ⋯⟩) (HSMul.hSMul c x) i) (HSMu …
    -/
    simp only [Pi.smul_apply, LinearEquiv.map_smul, Finsupp.coe_smul]
    /-
      🎉 no goals
    -/
    /-
      ι : Type u_1
      R : Type u_3
      M : Type u_6
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      b : Basis ι R M
      x : M
      i : ι
      a✝ : Nontrivial R
      ⊢ ∀ (i : ι), Eq ((fun x i => (b.reindexRange.repr x) ⟨b i, ⋯⟩) (b i)) ⇑(Finsup …
    -/
  · intro i
    /-
      ι : Type u_1
      R : Type u_3
      M : Type u_6
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      b : Basis ι R M
      x : M
      i✝ : ι
      a✝ : Nontrivial R
      i : ι
      ⊢ Eq ((fun x i => (b.reindexRange.repr x) ⟨b i, ⋯⟩) (b i)) ⇑(Finsupp.single i 1)
    -/
    ext j
    /-
      case h
      ι : Type u_1
      R : Type u_3
      M : Type u_6
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      b : Basis ι R M
      x : M
      i✝ : ι
      a✝ : Nontrivial R
      i j : ι
      ⊢ Eq ((fun x i => (b.reindexRange.repr x) ⟨b i, ⋯⟩) (b i) j) ((Finsupp.single  …
    -/
    simp only [reindexRange_repr_self]
    /-
      case h
      ι : Type u_1
      R : Type u_3
      M : Type u_6
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      b : Basis ι R M
      x : M
      i✝ : ι
      a✝ : Nontrivial R
      i j : ι
      ⊢ Eq ((Finsupp.single ⟨b i, ⋯⟩ 1) ⟨b j, ⋯⟩) ((Finsupp.single i 1) j)
    -/
    apply Finsupp.single_apply_left (f := fun i => (⟨b i, _⟩ : Set.range b))
    /-
      case h.hf
      ι : Type u_1
      R : Type u_3
      M : Type u_6
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      b : Basis ι R M
      x : M
      i✝ : ι
      a✝ : Nontrivial R
      i j : ι
      ⊢ Function.Injective fun i => ⟨b i, ⋯⟩
    -/
    exact fun i j h => b.injective (Subtype.mk.inj h)
    /-
      🎉 no goals
    -/


@[simp]
theorem reindexRange_repr (x : M) (i : ι) (h := Set.mem_range_self i) :
    b.reindexRange.repr x ⟨b i, h⟩ = b.repr x i :=
  b.reindexRange_repr' _ rfl


/-- `b.reindexFinsetRange` is a basis indexed by `Finset.univ.image b`,
the finite set of basis vectors themselves. -/
def reindexFinsetRange : Basis (Finset.univ.image b) R M :=
                                                          /-
                                                            ι : Type u_1
                                                            ι' : Type u_2
                                                            R : Type u_3
                                                            R₂ : Type u_4
                                                            K : Type u_5
                                                            M : Type u_6
                                                            M' : Type u_7
                                                            M'' : Type u_8
                                                            V : Type u
                                                            V' : Type u_9
                                                            inst✝⁶ : Semiring R
                                                            inst✝⁵ : AddCommMonoid M
                                                            inst✝⁴ : Module R M
                                                            inst✝³ : AddCommMonoid M'
                                                            inst✝² : Module R M'
                                                            b b₁ : Basis ι R M
                                                            i : ι
                                                            c : R
                                                            x : M
                                                            b' : Basis ι' R M'
                                                            e : Equiv ι ι'
                                                            inst✝¹ : Fintype ι
                                                            inst✝ : DecidableEq M
                                                            ⊢ ∀ (a : M), Iff (Membership.mem (Set.range ⇑b) a) (Membership.mem (Finset.ima …
                                                          -/
  b.reindexRange.reindex ((Equiv.refl M).subtypeEquiv (by simp))
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem reindexFinsetRange_self (i : ι) (h := Finset.mem_image_of_mem b (Finset.mem_univ i)) :
    b.reindexFinsetRange ⟨b i, h⟩ = b i := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    b : Basis ι R M
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq M
    i : ι
    h : optParam (Membership.mem (Finset.image (⇑b) Finset.univ) (b i)) ⋯
    ⊢ Eq (b.reindexFinsetRange ⟨b i, h⟩) (b i)
  -/
  rw [reindexFinsetRange, reindex_apply, reindexRange_apply]
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    b : Basis ι R M
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq M
    i : ι
    h : optParam (Membership.mem (Finset.image (⇑b) Finset.univ) (b i)) ⋯
    ⊢ Eq (↑(((Equiv.refl M).subtypeEquiv ⋯).symm ⟨b i, h⟩)) (b i)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem reindexFinsetRange_apply (x : Finset.univ.image b) : b.reindexFinsetRange x = x := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    b : Basis ι R M
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq M
    x : Subtype fun x => Membership.mem (Finset.image (⇑b) Finset.univ) x
    ⊢ Eq (b.reindexFinsetRange x) ↑x
  -/
  rcases x with ⟨bi, hbi⟩
  /-
    case mk
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    b : Basis ι R M
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq M
    bi : M
    hbi : Membership.mem (Finset.image (⇑b) Finset.univ) bi
    ⊢ Eq (b.reindexFinsetRange ⟨bi, hbi⟩) ↑⟨bi, hbi⟩
  -/
  rcases Finset.mem_image.mp hbi with ⟨i, -, rfl⟩
  /-
    case mk.intro.intro
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    b : Basis ι R M
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq M
    i : ι
    hbi : Membership.mem (Finset.image (⇑b) Finset.univ) (b i)
    ⊢ Eq (b.reindexFinsetRange ⟨b i, hbi⟩) ↑⟨b i, hbi⟩
  -/
  exact b.reindexFinsetRange_self i
  /-
    🎉 no goals
  -/


theorem reindexFinsetRange_repr_self (i : ι) :
    b.reindexFinsetRange.repr (b i) =
      Finsupp.single ⟨b i, Finset.mem_image_of_mem b (Finset.mem_univ i)⟩ 1 := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    b : Basis ι R M
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq M
    i : ι
    ⊢ Eq (b.reindexFinsetRange.repr (b i)) (Finsupp.single ⟨b i, ⋯⟩ 1)
  -/
  ext ⟨bi, hbi⟩
  /-
    case h.mk
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    b : Basis ι R M
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq M
    i : ι
    bi : M
    hbi : Membership.mem (Finset.image (⇑b) Finset.univ) bi
    ⊢ Eq ((b.reindexFinsetRange.repr (b i)) ⟨bi, hbi⟩) ((Finsupp.single ⟨b i, ⋯⟩ 1 …
  -/
  rw [reindexFinsetRange, repr_reindex, Finsupp.mapDomain_equiv_apply, reindexRange_repr_self]
  -- Porting note: replaced a `convert; refl` with `simp`
  /-
    case h.mk
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    b : Basis ι R M
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq M
    i : ι
    bi : M
    hbi : Membership.mem (Finset.image (⇑b) Finset.univ) bi
    ⊢ Eq ((Finsupp.single ⟨b i, ⋯⟩ 1) (((Equiv.refl M).subtypeEquiv ⋯).symm ⟨bi, h …
  -/
  simp [Finsupp.single_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem reindexFinsetRange_repr (x : M) (i : ι)
    (h := Finset.mem_image_of_mem b (Finset.mem_univ i)) :
                                                            /-
                                                              ι : Type u_1
                                                              R : Type u_3
                                                              M : Type u_6
                                                              inst✝⁴ : Semiring R
                                                              inst✝³ : AddCommMonoid M
                                                              inst✝² : Module R M
                                                              b : Basis ι R M
                                                              inst✝¹ : Fintype ι
                                                              inst✝ : DecidableEq M
                                                              x : M
                                                              i : ι
                                                              h : optParam (Membership.mem (Finset.image (⇑b) Finset.univ) (b i)) ⋯
                                                              ⊢ Eq ((b.reindexFinsetRange.repr x) ⟨b i, h⟩) ((b.repr x) i)
                                                            -/
    b.reindexFinsetRange.repr x ⟨b i, h⟩ = b.repr x i := by simp [reindexFinsetRange]
                                                            /-
                                                              🎉 no goals
                                                            -/


protected theorem mem_span (x : M) : x ∈ span R (range b) :=
  span_mono (image_subset_range _ _) (mem_span_repr_support b x)


@[simp]
protected theorem span_eq : span R (range b) = ⊤ :=
  eq_top_iff.mpr fun x _ => b.mem_span x


theorem index_nonempty (b : Basis ι R M) [Nontrivial M] : Nonempty ι := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    b : Basis ι R M
    inst✝ : Nontrivial M
    ⊢ Nonempty ι
  -/
  obtain ⟨x, y, ne⟩ : ∃ x y : M, x ≠ y := Nontrivial.exists_pair_ne
  /-
    case intro.intro
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    b : Basis ι R M
    inst✝ : Nontrivial M
    x y : M
    ne : Ne x y
    ⊢ Nonempty ι
  -/
  obtain ⟨i, _⟩ := not_forall.mp (mt b.ext_elem_iff.2 ne)
  /-
    case intro.intro.intro
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    b : Basis ι R M
    inst✝ : Nontrivial M
    x y : M
    ne : Ne x y
    i : ι
    h✝ : Not (Eq ((b.repr x) i) ((b.repr y) i))
    ⊢ Nonempty ι
  -/
  exact ⟨i⟩
  /-
    🎉 no goals
  -/


/-- If the submodule `P` has a basis, `x ∈ P` iff it is a linear combination of basis vectors. -/
theorem mem_submodule_iff {P : Submodule R M} (b : Basis ι R P) {x : M} :
    x ∈ P ↔ ∃ c : ι →₀ R, x = Finsupp.sum c fun i x => x • (b i : M) := by
  conv_lhs =>
    rw [← P.range_subtype, ← Submodule.map_top, ← b.span_eq, Submodule.map_span, ← Set.range_comp,
        ← Finsupp.range_linearCombination]
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    P : Submodule R M
    b : Basis ι R (Subtype fun x => Membership.mem P x)
    x : M
    ⊢ Iff (Membership.mem (LinearMap.range (Finsupp.linearCombination R (Function. …
  -/
  simp [@eq_comm _ x, Function.comp, Finsupp.linearCombination_apply]
  /-
    🎉 no goals
  -/


/-- Construct a linear map given the value at the basis, called `Basis.constr b S f` where `b` is
a basis, `f` is the value of the linear map over the elements of the basis, and `S` is an
extra semiring (typically `S = R` or `S = ℕ`).

This definition is parameterized over an extra `Semiring S`,
such that `SMulCommClass R S M'` holds.
If `R` is commutative, you can set `S := R`; if `R` is not commutative,
you can recover an `AddEquiv` by setting `S := ℕ`.
See library note [bundled maps over different rings].
-/
def constr : (ι → M') ≃ₗ[S] M →ₗ[R] M' where
  toFun f := (Finsupp.linearCombination R id).comp <| Finsupp.lmapDomain R R f ∘ₗ ↑b.repr
  invFun f i := f (b i)
  left_inv f := by
    /-
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      K : Type u_5
      M : Type u_6
      M' : Type u_7
      M'' : Type u_8
      V : Type u
      V' : Type u_9
      inst✝⁷ : Semiring R
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : Module R M'
      b b₁ : Basis ι R M
      i : ι
      c : R
      x : M
      S : Type u_10
      inst✝² : Semiring S
      inst✝¹ : Module S M'
      inst✝ : SMulCommClass R S M'
      f : ι → M'
      ⊢ Eq ((fun f i => f (b i)) ({ toFun := fun f => (Finsupp.linearCombination R i …
    -/
    ext
    /-
      case h
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      K : Type u_5
      M : Type u_6
      M' : Type u_7
      M'' : Type u_8
      V : Type u
      V' : Type u_9
      inst✝⁷ : Semiring R
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : Module R M'
      b b₁ : Basis ι R M
      i : ι
      c : R
      x : M
      S : Type u_10
      inst✝² : Semiring S
      inst✝¹ : Module S M'
      inst✝ : SMulCommClass R S M'
      f : ι → M'
      x✝ : ι
      ⊢ Eq ((fun f i => f (b i)) ({ toFun := fun f => (Finsupp.linearCombination R i …
    -/
    simp
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      K : Type u_5
      M : Type u_6
      M' : Type u_7
      M'' : Type u_8
      V : Type u
      V' : Type u_9
      inst✝⁷ : Semiring R
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : Module R M'
      b b₁ : Basis ι R M
      i : ι
      c : R
      x : M
      S : Type u_10
      inst✝² : Semiring S
      inst✝¹ : Module S M'
      inst✝ : SMulCommClass R S M'
      f g : ι → M'
      ⊢ Eq ((fun f => (Finsupp.linearCombination R id).comp ((Finsupp.lmapDomain R R …
    -/
    /-
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      K : Type u_5
      M : Type u_6
      M' : Type u_7
      M'' : Type u_8
      V : Type u
      V' : Type u_9
      inst✝⁷ : Semiring R
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : Module R M'
      b b₁ : Basis ι R M
      i : ι
      c : R
      x : M
      S : Type u_10
      inst✝² : Semiring S
      inst✝¹ : Module S M'
      inst✝ : SMulCommClass R S M'
      f : LinearMap (RingHom.id R) M M'
      ⊢ Eq ({ toFun := fun f => (Finsupp.linearCombination R id).comp ((Finsupp.lmap …
    -/
    /-
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      K : Type u_5
      M : Type u_6
      M' : Type u_7
      M'' : Type u_8
      V : Type u
      V' : Type u_9
      inst✝⁷ : Semiring R
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : Module R M'
      b b₁ : Basis ι R M
      i✝ : ι
      c : R
      x : M
      S : Type u_10
      inst✝² : Semiring S
      inst✝¹ : Module S M'
      inst✝ : SMulCommClass R S M'
      f g : ι → M'
      i : ι
      ⊢ Eq (((fun f => (Finsupp.linearCombination R id).comp ((Finsupp.lmapDomain R  …
    -/
    refine b.ext fun i => ?_
    /-
      🎉 no goals
    -/
    /-
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      K : Type u_5
      M : Type u_6
      M' : Type u_7
      M'' : Type u_8
      V : Type u
      V' : Type u_9
      inst✝⁷ : Semiring R
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : Module R M'
      b b₁ : Basis ι R M
      i✝ : ι
      c : R
      x : M
      S : Type u_10
      inst✝² : Semiring S
      inst✝¹ : Module S M'
      inst✝ : SMulCommClass R S M'
      f : LinearMap (RingHom.id R) M M'
      i : ι
      ⊢ Eq (({ toFun := fun f => (Finsupp.linearCombination R id).comp ((Finsupp.lma …
    -/
    /-
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      K : Type u_5
      M : Type u_6
      M' : Type u_7
      M'' : Type u_8
      V : Type u
      V' : Type u_9
      inst✝⁷ : Semiring R
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : Module R M'
      b b₁ : Basis ι R M
      i : ι
      c✝ : R
      x : M
      S : Type u_10
      inst✝² : Semiring S
      inst✝¹ : Module S M'
      inst✝ : SMulCommClass R S M'
      c : S
      f : ι → M'
      ⊢ Eq ({ toFun := fun f => (Finsupp.linearCombination R id).comp ((Finsupp.lmap …
    -/
    simp
    /-
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      K : Type u_5
      M : Type u_6
      M' : Type u_7
      M'' : Type u_8
      V : Type u
      V' : Type u_9
      inst✝⁷ : Semiring R
      inst✝⁶ : AddCommMonoid M
      inst✝⁵ : Module R M
      inst✝⁴ : AddCommMonoid M'
      inst✝³ : Module R M'
      b b₁ : Basis ι R M
      i✝ : ι
      c✝ : R
      x : M
      S : Type u_10
      inst✝² : Semiring S
      inst✝¹ : Module S M'
      inst✝ : SMulCommClass R S M'
      c : S
      f : ι → M'
      i : ι
      ⊢ Eq (({ toFun := fun f => (Finsupp.linearCombination R id).comp ((Finsupp.lma …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  map_add' f g := by
    refine b.ext fun i => ?_
    simp
  map_smul' c f := by
    refine b.ext fun i => ?_
    simp


theorem constr_def (f : ι → M') :
    constr (M' := M') b S f = linearCombination R id ∘ₗ Finsupp.lmapDomain R R f ∘ₗ ↑b.repr :=
  rfl


theorem constr_apply (f : ι → M') (x : M) :
    constr (M' := M') b S f x = (b.repr x).sum fun b a => a • f b := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    M' : Type u_7
    inst✝⁷ : Semiring R
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommMonoid M'
    inst✝³ : Module R M'
    b : Basis ι R M
    S : Type u_10
    inst✝² : Semiring S
    inst✝¹ : Module S M'
    inst✝ : SMulCommClass R S M'
    f : ι → M'
    x : M
    ⊢ Eq (((b.constr S) f) x) ((b.repr x).sum fun b a => HSMul.hSMul a (f b))
  -/
  simp only [constr_def, LinearMap.comp_apply, lmapDomain_apply, linearCombination_apply]
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    M' : Type u_7
    inst✝⁷ : Semiring R
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommMonoid M'
    inst✝³ : Module R M'
    b : Basis ι R M
    S : Type u_10
    inst✝² : Semiring S
    inst✝¹ : Module S M'
    inst✝ : SMulCommClass R S M'
    f : ι → M'
    x : M
    ⊢ Eq ((Finsupp.mapDomain f (↑b.repr x)).sum fun i a => HSMul.hSMul a (id i)) ( …
  -/
                                       /-
                                         🎉 no goals
                                       -/
                                       /-
                                         🎉 no goals
                                       -/
  rw [Finsupp.sum_mapDomain_index] <;> simp [add_smul]
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem constr_basis (f : ι → M') (i : ι) : (constr (M' := M') b S f : M → M') (b i) = f i := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    M' : Type u_7
    inst✝⁷ : Semiring R
    inst✝⁶ : AddCommMonoid M
    inst✝⁵ : Module R M
    inst✝⁴ : AddCommMonoid M'
    inst✝³ : Module R M'
    b : Basis ι R M
    S : Type u_10
    inst✝² : Semiring S
    inst✝¹ : Module S M'
    inst✝ : SMulCommClass R S M'
    f : ι → M'
    i : ι
    ⊢ Eq (((b.constr S) f) (b i)) (f i)
  -/
  simp [Basis.constr_apply, b.repr_self]
  /-
    🎉 no goals
  -/


theorem constr_eq {g : ι → M'} {f : M →ₗ[R] M'} (h : ∀ i, g i = f (b i)) :
    constr (M' := M') b S g = f :=
  b.ext fun i => (b.constr_basis S g i).trans (h i)


theorem constr_self (f : M →ₗ[R] M') : (constr (M' := M') b S fun i => f (b i)) = f :=
  b.constr_eq S fun _ => rfl


theorem constr_range {f : ι → M'} :
    LinearMap.range (constr (M' := M') b S f) = span R (range f) := by
  rw [b.constr_def S f, LinearMap.range_comp, LinearMap.range_comp, LinearEquiv.range, ←
    Finsupp.supported_univ, Finsupp.lmapDomain_supported, ← Set.image_univ, ←
    Finsupp.span_image_eq_map_linearCombination, Set.image_id]


@[simp]
theorem constr_comp (f : M' →ₗ[R] M') (v : ι → M') :
    constr (M' := M') b S (f ∘ v) = f.comp (constr (M' := M') b S v) :=
                    /-
                      ι : Type u_1
                      R : Type u_3
                      M : Type u_6
                      M' : Type u_7
                      inst✝⁷ : Semiring R
                      inst✝⁶ : AddCommMonoid M
                      inst✝⁵ : Module R M
                      inst✝⁴ : AddCommMonoid M'
                      inst✝³ : Module R M'
                      b : Basis ι R M
                      S : Type u_10
                      inst✝² : Semiring S
                      inst✝¹ : Module S M'
                      inst✝ : SMulCommClass R S M'
                      f : LinearMap (RingHom.id R) M' M'
                      v : ι → M'
                      i : ι
                      ⊢ Eq (((b.constr S) (Function.comp (⇑f) v)) (b i)) ((f.comp ((b.constr S) v))  …
                    -/
  b.ext fun i => by simp only [Basis.constr_basis, LinearMap.comp_apply, Function.comp]
                    /-
                      🎉 no goals
                    -/


/-- If `b` is a basis for `M` and `b'` a basis for `M'`, and the index types are equivalent,
`b.equiv b' e` is a linear equivalence `M ≃ₗ[R] M'`, mapping `b i` to `b' (e i)`. -/
protected def equiv : M ≃ₗ[R] M' :=
  b.repr.trans (b'.reindex e.symm).repr.symm


@[simp]
                                                          /-
                                                            ι : Type u_1
                                                            ι' : Type u_2
                                                            R : Type u_3
                                                            M : Type u_6
                                                            M' : Type u_7
                                                            inst✝⁴ : Semiring R
                                                            inst✝³ : AddCommMonoid M
                                                            inst✝² : Module R M
                                                            inst✝¹ : AddCommMonoid M'
                                                            inst✝ : Module R M'
                                                            b : Basis ι R M
                                                            i : ι
                                                            b' : Basis ι' R M'
                                                            e : Equiv ι ι'
                                                            ⊢ Eq ((b.equiv b' e) (b i)) (b' (e i))
                                                          -/
theorem equiv_apply : b.equiv b' e (b i) = b' (e i) := by simp [Basis.equiv]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
theorem equiv_refl : b.equiv b (Equiv.refl ι) = LinearEquiv.refl R M :=
                     /-
                       ι : Type u_1
                       R : Type u_3
                       M : Type u_6
                       inst✝² : Semiring R
                       inst✝¹ : AddCommMonoid M
                       inst✝ : Module R M
                       b : Basis ι R M
                       i : ι
                       ⊢ Eq ((b.equiv b (Equiv.refl ι)) (b i)) ((LinearEquiv.refl R M) (b i))
                     -/
  b.ext' fun i => by simp
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem equiv_symm : (b.equiv b' e).symm = b'.equiv b e.symm :=
                                                /-
                                                  ι : Type u_1
                                                  ι' : Type u_2
                                                  R : Type u_3
                                                  M : Type u_6
                                                  M' : Type u_7
                                                  inst✝⁴ : Semiring R
                                                  inst✝³ : AddCommMonoid M
                                                  inst✝² : Module R M
                                                  inst✝¹ : AddCommMonoid M'
                                                  inst✝ : Module R M'
                                                  b : Basis ι R M
                                                  b' : Basis ι' R M'
                                                  e : Equiv ι ι'
                                                  i : ι'
                                                  ⊢ Eq ((b.equiv b' e) ((b.equiv b' e).symm (b' i))) ((b.equiv b' e) ((b'.equiv  …
                                                -/
  b'.ext' fun i => (b.equiv b' e).injective (by simp)
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem equiv_trans {ι'' : Type*} (b'' : Basis ι'' R M'') (e : ι ≃ ι') (e' : ι' ≃ ι'') :
    (b.equiv b' e).trans (b'.equiv b'' e') = b.equiv b'' (e.trans e') :=
                     /-
                       ι : Type u_1
                       ι' : Type u_2
                       R : Type u_3
                       M : Type u_6
                       M' : Type u_7
                       M'' : Type u_8
                       inst✝⁶ : Semiring R
                       inst✝⁵ : AddCommMonoid M
                       inst✝⁴ : Module R M
                       inst✝³ : AddCommMonoid M'
                       inst✝² : Module R M'
                       b : Basis ι R M
                       b' : Basis ι' R M'
                       inst✝¹ : AddCommMonoid M''
                       inst✝ : Module R M''
                       ι'' : Type u_10
                       b'' : Basis ι'' R M''
                       e : Equiv ι ι'
                       e' : Equiv ι' ι''
                       i : ι
                       ⊢ Eq (((b.equiv b' e).trans (b'.equiv b'' e')) (b i)) ((b.equiv b'' (e.trans e …
                     -/
  b.ext' fun i => by simp
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem map_equiv (b : Basis ι R M) (b' : Basis ι' R M') (e : ι ≃ ι') :
    b.map (b.equiv b' e) = b'.reindex e.symm := by
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    M : Type u_6
    M' : Type u_7
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    b : Basis ι R M
    b' : Basis ι' R M'
    e : Equiv ι ι'
    ⊢ Eq (b.map (b.equiv b' e)) (b'.reindex e.symm)
  -/
  ext i
  /-
    case a
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    M : Type u_6
    M' : Type u_7
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    b : Basis ι R M
    b' : Basis ι' R M'
    e : Equiv ι ι'
    i : ι
    ⊢ Eq ((b.map (b.equiv b' e)) i) ((b'.reindex e.symm) i)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- `Basis.singleton ι R` is the basis sending the unique element of `ι` to `1 : R`. -/
protected def singleton (ι R : Type*) [Unique ι] [Semiring R] : Basis ι R R :=
  ofRepr
    { toFun := fun x => Finsupp.single default x
      invFun := fun f => f default
                              /-
                                ι✝ : Type u_1
                                ι' : Type u_2
                                R✝ : Type u_3
                                R₂ : Type u_4
                                K : Type u_5
                                M : Type u_6
                                M' : Type u_7
                                M'' : Type u_8
                                V : Type u
                                V' : Type u_9
                                inst✝⁶ : Semiring R✝
                                inst✝⁵ : AddCommMonoid M
                                inst✝⁴ : Module R✝ M
                                inst✝³ : AddCommMonoid M'
                                inst✝² : Module R✝ M'
                                b b₁ : Basis ι✝ R✝ M
                                i : ι✝
                                c : R✝
                                x✝ : M
                                ι : Type u_10
                                R : Type u_11
                                inst✝¹ : Unique ι
                                inst✝ : Semiring R
                                x : R
                                ⊢ Eq ((fun f => f Inhabited.default) ({ toFun := fun x => Finsupp.single Inhab …
                              -/
      left_inv := fun x => by simp
                                /-
                                  ι✝ : Type u_1
                                  ι' : Type u_2
                                  R✝ : Type u_3
                                  R₂ : Type u_4
                                  K : Type u_5
                                  M : Type u_6
                                  M' : Type u_7
                                  M'' : Type u_8
                                  V : Type u
                                  V' : Type u_9
                                  inst✝⁶ : Semiring R✝
                                  inst✝⁵ : AddCommMonoid M
                                  inst✝⁴ : Module R✝ M
                                  inst✝³ : AddCommMonoid M'
                                  inst✝² : Module R✝ M'
                                  b b₁ : Basis ι✝ R✝ M
                                  i : ι✝
                                  c : R✝
                                  x✝ : M
                                  ι : Type u_10
                                  R : Type u_11
                                  inst✝¹ : Unique ι
                                  inst✝ : Semiring R
                                  x y : R
                                  ⊢ Eq ((fun x => Finsupp.single Inhabited.default x) (HAdd.hAdd x y)) (HAdd.hAd …
                                -/
                              /-
                                🎉 no goals
                              -/
                                /-
                                  🎉 no goals
                                -/
                                 /-
                                   ι✝ : Type u_1
                                   ι' : Type u_2
                                   R✝ : Type u_3
                                   R₂ : Type u_4
                                   K : Type u_5
                                   M : Type u_6
                                   M' : Type u_7
                                   M'' : Type u_8
                                   V : Type u
                                   V' : Type u_9
                                   inst✝⁶ : Semiring R✝
                                   inst✝⁵ : AddCommMonoid M
                                   inst✝⁴ : Module R✝ M
                                   inst✝³ : AddCommMonoid M'
                                   inst✝² : Module R✝ M'
                                   b b₁ : Basis ι✝ R✝ M
                                   i : ι✝
                                   c✝ : R✝
                                   x✝ : M
                                   ι : Type u_10
                                   R : Type u_11
                                   inst✝¹ : Unique ι
                                   inst✝ : Semiring R
                                   c x : R
                                   ⊢ Eq ({ toFun := fun x => Finsupp.single Inhabited.default x, map_add' := ⋯ }. …
                                 -/
                                                   /-
                                                     ι✝ : Type u_1
                                                     ι' : Type u_2
                                                     R✝ : Type u_3
                                                     R₂ : Type u_4
                                                     K : Type u_5
                                                     M : Type u_6
                                                     M' : Type u_7
                                                     M'' : Type u_8
                                                     V : Type u
                                                     V' : Type u_9
                                                     inst✝⁶ : Semiring R✝
                                                     inst✝⁵ : AddCommMonoid M
                                                     inst✝⁴ : Module R✝ M
                                                     inst✝³ : AddCommMonoid M'
                                                     inst✝² : Module R✝ M'
                                                     b b₁ : Basis ι✝ R✝ M
                                                     i : ι✝
                                                     c : R✝
                                                     x : M
                                                     ι : Type u_10
                                                     R : Type u_11
                                                     inst✝¹ : Unique ι
                                                     inst✝ : Semiring R
                                                     f : Finsupp ι R
                                                     ⊢ Eq (({ toFun := fun x => Finsupp.single Inhabited.default x, map_add' := ⋯,  …
                                                   -/
                                 /-
                                   🎉 no goals
                                 -/
      right_inv := fun f => Finsupp.unique_ext (by simp)
                                                   /-
                                                     🎉 no goals
                                                   -/
      map_add' := fun x y => by simp
      map_smul' := fun c x => by simp }


@[simp]
theorem singleton_apply (ι R : Type*) [Unique ι] [Semiring R] (i) : Basis.singleton ι R i = 1 :=
                       /-
                         ι : Type u_10
                         R : Type u_11
                         inst✝¹ : Unique ι
                         inst✝ : Semiring R
                         i : ι
                         ⊢ Eq ((Basis.singleton ι R).repr 1) (Finsupp.single i 1)
                       -/
  apply_eq_iff.mpr (by simp [Basis.singleton])
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem singleton_repr (ι R : Type*) [Unique ι] [Semiring R] (x i) :
                                             /-
                                               ι : Type u_10
                                               R : Type u_11
                                               inst✝¹ : Unique ι
                                               inst✝ : Semiring R
                                               x : R
                                               i : ι
                                               ⊢ Eq (((Basis.singleton ι R).repr x) i) x
                                             -/
    (Basis.singleton ι R).repr x i = x := by simp [Basis.singleton, Unique.eq_default i]
                                             /-
                                               🎉 no goals
                                             -/


/-- If `M` is a subsingleton and `ι` is empty, this is the unique `ι`-indexed basis for `M`. -/
protected def empty [Subsingleton M] [IsEmpty ι] : Basis ι R M :=
  ofRepr 0


instance emptyUnique [Subsingleton M] [IsEmpty ι] : Unique (Basis ι R M) where
  default := Basis.empty M
  uniq := fun _ => congr_arg ofRepr <| Subsingleton.elim _ _


/-- A module over `R` with a finite basis is linearly equivalent to functions from its basis to `R`.
-/
def Basis.equivFun [Finite ι] (b : Basis ι R M) : M ≃ₗ[R] ι → R :=
  LinearEquiv.trans b.repr
    ({ Finsupp.equivFunOnFinite with
        toFun := (↑)
        map_add' := Finsupp.coe_add
        map_smul' := Finsupp.coe_smul } :
      (ι →₀ R) ≃ₗ[R] ι → R)


/-- A module over a finite ring that admits a finite basis is finite. -/
def Module.fintypeOfFintype [Fintype ι] (b : Basis ι R M) [Fintype R] : Fintype M :=
  haveI := Classical.decEq ι
  Fintype.ofEquiv _ b.equivFun.toEquiv.symm


theorem Module.card_fintype [Fintype ι] (b : Basis ι R M) [Fintype R] [Fintype M] :
    card M = card R ^ card ι := by
  classical
    calc
      card M = card (ι → R) := card_congr b.equivFun.toEquiv
      _ = card R ^ card ι := card_fun


/-- Given a basis `v` indexed by `ι`, the canonical linear equivalence between `ι → R` and `M` maps
a function `x : ι → R` to the linear combination `∑_i x i • v i`. -/
@[simp]
theorem Basis.equivFun_symm_apply [Fintype ι] (b : Basis ι R M) (x : ι → R) :
    b.equivFun.symm x = ∑ i, x i • b i := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Fintype ι
    b : Basis ι R M
    x : ι → R
    ⊢ Eq (b.equivFun.symm x) (Finset.univ.sum fun i => HSMul.hSMul (x i) (b i))
  -/
  simp [Basis.equivFun, Finsupp.linearCombination_apply, sum_fintype, equivFunOnFinite]
  /-
    🎉 no goals
  -/


@[simp]
theorem Basis.equivFun_apply [Finite ι] (b : Basis ι R M) (u : M) : b.equivFun u = b.repr u :=
  rfl


@[simp]
theorem Basis.map_equivFun [Finite ι] (b : Basis ι R M) (f : M ≃ₗ[R] M') :
    (b.map f).equivFun = f.symm.trans b.equivFun :=
  rfl


theorem Basis.sum_equivFun [Fintype ι] (b : Basis ι R M) (u : M) :
    ∑ i, b.equivFun u i • b i = u := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Fintype ι
    b : Basis ι R M
    u : M
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (b.equivFun u i) (b i)) u
  -/
  rw [← b.equivFun_symm_apply, b.equivFun.symm_apply_apply]
  /-
    🎉 no goals
  -/


theorem Basis.sum_repr [Fintype ι] (b : Basis ι R M) (u : M) : ∑ i, b.repr u i • b i = u :=
  b.sum_equivFun u


@[simp]
theorem Basis.equivFun_self [Finite ι] [DecidableEq ι] (b : Basis ι R M) (i j : ι) :
                                                      /-
                                                        ι : Type u_1
                                                        R : Type u_3
                                                        M : Type u_6
                                                        inst✝⁴ : Semiring R
                                                        inst✝³ : AddCommMonoid M
                                                        inst✝² : Module R M
                                                        inst✝¹ : Finite ι
                                                        inst✝ : DecidableEq ι
                                                        b : Basis ι R M
                                                        i j : ι
                                                        ⊢ Eq (b.equivFun (b i) j) (ite (Eq i j) 1 0)
                                                      -/
    b.equivFun (b i) j = if i = j then 1 else 0 := by rw [b.equivFun_apply, b.repr_self_apply]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem Basis.repr_sum_self [Fintype ι] (b : Basis ι R M) (c : ι → R) :
    b.repr (∑ i, c i • b i) = c := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Fintype ι
    b : Basis ι R M
    c : ι → R
    ⊢ Eq (⇑(b.repr (Finset.univ.sum fun i => HSMul.hSMul (c i) (b i)))) c
  -/
  simp_rw [← b.equivFun_symm_apply, ← b.equivFun_apply, b.equivFun.apply_symm_apply]
  /-
    🎉 no goals
  -/


/-- Define a basis by mapping each vector `x : M` to its coordinates `e x : ι → R`,
as long as `ι` is finite. -/
def Basis.ofEquivFun [Finite ι] (e : M ≃ₗ[R] ι → R) : Basis ι R M :=
  .ofRepr <| e.trans <| LinearEquiv.symm <| Finsupp.linearEquivFunOnFinite R R ι


@[simp]
theorem Basis.ofEquivFun_repr_apply [Finite ι] (e : M ≃ₗ[R] ι → R) (x : M) (i : ι) :
    (Basis.ofEquivFun e).repr x i = e x i :=
  rfl


@[simp]
theorem Basis.coe_ofEquivFun [Finite ι] [DecidableEq ι] (e : M ≃ₗ[R] ι → R) :
    (Basis.ofEquivFun e : ι → M) = fun i => e.symm (Pi.single i 1) :=
  funext fun i =>
    e.injective <|
      funext fun j => by
        /-
          ι : Type u_1
          R : Type u_3
          M : Type u_6
          inst✝⁴ : Semiring R
          inst✝³ : AddCommMonoid M
          inst✝² : Module R M
          inst✝¹ : Finite ι
          inst✝ : DecidableEq ι
          e : LinearEquiv (RingHom.id R) M (ι → R)
          i j : ι
          ⊢ Eq (e ((Basis.ofEquivFun e) i) j) (e (e.symm (Pi.single i 1)) j)
        -/
        simp [Basis.ofEquivFun, ← Finsupp.single_eq_pi_single]
        /-
          🎉 no goals
        -/


@[simp]
theorem Basis.ofEquivFun_equivFun [Finite ι] (v : Basis ι R M) :
    Basis.ofEquivFun v.equivFun = v :=
                             /-
                               ι : Type u_1
                               R : Type u_3
                               M : Type u_6
                               inst✝³ : Semiring R
                               inst✝² : AddCommMonoid M
                               inst✝¹ : Module R M
                               inst✝ : Finite ι
                               v : Basis ι R M
                               ⊢ Eq (Basis.ofEquivFun v.equivFun).repr v.repr
                             -/
  Basis.repr_injective <| by ext; rfl
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem Basis.equivFun_ofEquivFun [Finite ι] (e : M ≃ₗ[R] ι → R) :
    (Basis.ofEquivFun e).equivFun = e := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Finite ι
    e : LinearEquiv (RingHom.id R) M (ι → R)
    ⊢ Eq (Basis.ofEquivFun e).equivFun e
  -/
  ext j
  /-
    case h.h
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Finite ι
    e : LinearEquiv (RingHom.id R) M (ι → R)
    j : M
    x✝ : ι
    ⊢ Eq ((Basis.ofEquivFun e).equivFun j x✝) (e j x✝)
  -/
  simp_rw [Basis.equivFun_apply, Basis.ofEquivFun_repr_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem Basis.constr_apply_fintype [Fintype ι] (b : Basis ι R M) (f : ι → M') (x : M) :
    (constr (M' := M') b S f : M → M') x = ∑ i, b.equivFun x i • f i := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    M' : Type u_7
    inst✝⁸ : Semiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Module R M
    inst✝⁵ : AddCommMonoid M'
    inst✝⁴ : Module R M'
    S : Type u_10
    inst✝³ : Semiring S
    inst✝² : Module S M'
    inst✝¹ : SMulCommClass R S M'
    inst✝ : Fintype ι
    b : Basis ι R M
    f : ι → M'
    x : M
    ⊢ Eq (((b.constr S) f) x) (Finset.univ.sum fun i => HSMul.hSMul (b.equivFun x  …
  -/
  simp [b.constr_apply, b.equivFun_apply, Finsupp.sum_fintype]
  /-
    🎉 no goals
  -/


/-- If the submodule `P` has a finite basis,
`x ∈ P` iff it is a linear combination of basis vectors. -/
theorem Basis.mem_submodule_iff' [Fintype ι] {P : Submodule R M} (b : Basis ι R P) {x : M} :
    x ∈ P ↔ ∃ c : ι → R, x = ∑ i, c i • (b i : M) :=
  b.mem_submodule_iff.trans <|
    Finsupp.equivFunOnFinite.exists_congr_left.trans <|
                               /-
                                 ι : Type u_1
                                 R : Type u_3
                                 M : Type u_6
                                 inst✝³ : Semiring R
                                 inst✝² : AddCommMonoid M
                                 inst✝¹ : Module R M
                                 inst✝ : Fintype ι
                                 P : Submodule R M
                                 b : Basis ι R (Subtype fun x => Membership.mem P x)
                                 x : M
                                 c : ι → R
                                 ⊢ Iff (Eq x ((Finsupp.equivFunOnFinite.symm c).sum fun i x => HSMul.hSMul x ↑( …
                               -/
      exists_congr fun c => by simp [Finsupp.sum_fintype, Finsupp.equivFunOnFinite]
                               /-
                                 🎉 no goals
                               -/


theorem Basis.coord_equivFun_symm [Finite ι] (b : Basis ι R M) (i : ι) (f : ι → R) :
    b.coord i (b.equivFun.symm f) = f i :=
  b.coord_repr_symm i (Finsupp.equivFunOnFinite.symm f)


/-- If `b` is a basis for `M` and `b'` a basis for `M'`,
and `f`, `g` form a bijection between the basis vectors,
`b.equiv' b' f g hf hg hgf hfg` is a linear equivalence `M ≃ₗ[R] M'`, mapping `b i` to `f (b i)`.
-/
def equiv' (f : M → M') (g : M' → M) (hf : ∀ i, f (b i) ∈ range b') (hg : ∀ i, g (b' i) ∈ range b)
    (hgf : ∀ i, g (f (b i)) = b i) (hfg : ∀ i, f (g (b' i)) = b' i) : M ≃ₗ[R] M' :=
  { constr (M' := M') b R (f ∘ b) with
    invFun := constr (M' := M) b' R (g ∘ b')
    left_inv :=
      have : (constr (M' := M) b' R (g ∘ b')).comp (constr (M' := M') b R (f ∘ b)) = LinearMap.id :=
        b.ext fun i =>
          Exists.elim (hf i) fun i' hi' => by
            rw [LinearMap.comp_apply, b.constr_basis, Function.comp_apply, ← hi', b'.constr_basis,
              Function.comp_apply, hi', hgf, LinearMap.id_apply]
      fun x => congr_arg (fun h : M →ₗ[R] M => h x) this
    right_inv :=
      have : (constr (M' := M') b R (f ∘ b)).comp (constr (M' := M) b' R (g ∘ b')) = LinearMap.id :=
        b'.ext fun i =>
          Exists.elim (hg i) fun i' hi' => by
            rw [LinearMap.comp_apply, b'.constr_basis, Function.comp_apply, ← hi', b.constr_basis,
              Function.comp_apply, hi', hfg, LinearMap.id_apply]
      fun x => congr_arg (fun h : M' →ₗ[R] M' => h x) this }


@[simp]
theorem equiv'_apply (f : M → M') (g : M' → M) (hf hg hgf hfg) (i : ι) :
    b.equiv' b' f g hf hg hgf hfg (b i) = f (b i) :=
  b.constr_basis R _ _


@[simp]
theorem equiv'_symm_apply (f : M → M') (g : M' → M) (hf hg hgf hfg) (i : ι') :
    (b.equiv' b' f g hf hg hgf hfg).symm (b' i) = g (b' i) :=
  b'.constr_basis R _ _


theorem sum_repr_mul_repr {ι'} [Fintype ι'] (b' : Basis ι' R M) (x : M) (i : ι) :
    (∑ j : ι', b.repr (b' j) i * b'.repr x j) = b.repr x i := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    b : Basis ι R M
    ι' : Type u_10
    inst✝ : Fintype ι'
    b' : Basis ι' R M
    x : M
    i : ι
    ⊢ Eq (Finset.univ.sum fun j => HMul.hMul ((b.repr (b' j)) i) ((b'.repr x) j))  …
  -/
  conv_rhs => rw [← b'.sum_repr x]
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    b : Basis ι R M
    ι' : Type u_10
    inst✝ : Fintype ι'
    b' : Basis ι' R M
    x : M
    i : ι
    ⊢ Eq (Finset.univ.sum fun j => HMul.hMul ((b.repr (b' j)) i) ((b'.repr x) j))  …
  -/
  simp_rw [map_sum, map_smul, Finset.sum_apply']
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    b : Basis ι R M
    ι' : Type u_10
    inst✝ : Fintype ι'
    b' : Basis ι' R M
    x : M
    i : ι
    ⊢ Eq (Finset.univ.sum fun j => HMul.hMul ((b.repr (b' j)) i) ((b'.repr x) j))  …
  -/
  refine Finset.sum_congr rfl fun j _ => ?_
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_6
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    b : Basis ι R M
    ι' : Type u_10
    inst✝ : Fintype ι'
    b' : Basis ι' R M
    x : M
    i : ι
    j : ι'
    x✝ : Membership.mem Finset.univ j
    ⊢ Eq (HMul.hMul ((b.repr (b' j)) i) ((b'.repr x) j)) ((HSMul.hSMul ((b'.repr x …
  -/
  rw [Finsupp.smul_apply, smul_eq_mul, mul_comm]
  /-
    🎉 no goals
  -/



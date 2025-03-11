/-- The proposition that the relation $\sum_i f_i x_i = 0$ in $M$ is trivial.
That is, there exist a finite index type $\kappa$, elements
$(y_j)_{j \in \kappa}$ of $M$, and elements $(a_{ij})_{i \in \iota, j \in \kappa}$ of $R$
such that for all $i$,
$$x_i = \sum_j a_{ij} y_j$$
and for all $j$,
$$\sum_{i} f_i a_{ij} = 0.$$
By `Module.sum_smul_eq_zero_of_isTrivialRelation`, this condition implies $\sum_i f_i x_i = 0$. -/
abbrev IsTrivialRelation : Prop :=
  ∃ (κ : Type u) (_ : Fintype κ) (a : ι → κ → R) (y : κ → M),
    (∀ i, x i = ∑ j, a i j • y j) ∧ ∀ j, ∑ i, f i * a i j = 0


/-- `Module.IsTrivialRelation` is equivalent to the predicate `TensorProduct.VanishesTrivially`
defined in `Mathlib/LinearAlgebra/TensorProduct/Vanishing.lean`. -/
theorem isTrivialRelation_iff_vanishesTrivially :
    IsTrivialRelation f x ↔ VanishesTrivially R f x := by
  /-
    R M : Type u
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    ι : Type u
    inst✝ : Fintype ι
    f : ι → R
    x : ι → M
    ⊢ Iff (Module.IsTrivialRelation f x) (TensorProduct.VanishesTrivially R f x)
  -/
  simp only [IsTrivialRelation, VanishesTrivially, smul_eq_mul, mul_comm]
  /-
    🎉 no goals
  -/



/-- If the relation given by $(f_i)_{i \in \iota}$ and $(x_i)_{i \in \iota}$ is trivial, then
$\sum_{i} f_i x_i$ is actually equal to $0$. -/
theorem sum_smul_eq_zero_of_isTrivialRelation (h : IsTrivialRelation f x) :
    ∑ i, f i • x i = 0 := by
  simpa using
    congr_arg (TensorProduct.lid R M) <|
      sum_tmul_eq_zero_of_vanishesTrivially R (isTrivialRelation_iff_vanishesTrivially.mp h)


variable (R M) in
/-- **Equational criterion for flatness** [Stacks 00HK](https://stacks.math.columbia.edu/tag/00HK),
combined form.

Let $M$ be a module over a commutative ring $R$. The following are equivalent:
* $M$ is flat.
* For all ideals $I \subseteq R$, the map $I \otimes M \to M$ is injective.
* Every $\sum_i f_i \otimes x_i$ that vanishes in $R \otimes M$ vanishes trivially.
* Every relation $\sum_i f_i x_i = 0$ in $M$ is trivial.
* For all finite index types $\iota$, all elements $f \in R^{\iota}$, and all homomorphisms
$x \colon R^{\iota} \to M$ such that $x(f) = 0$, there exist a finite index type $\kappa$ and module
homomorphisms $a \colon R^{\iota} \to R^{\kappa}$ and $y \colon R^{\kappa} \to M$ such
that $x = y \circ a$ and $a(f) = 0$.
* For all finite free modules $N$, all elements $f \in N$, and all homomorphisms $x \colon N \to M$
such that $x(f) = 0$, there exist a finite index type $\kappa$ and module homomorphisms
$a \colon N \to R^{\kappa}$ and $y \colon R^{\kappa} \to M$ such that $x = y \circ a$ and
$a(f) = 0$. -/
theorem tfae_equational_criterion : List.TFAE [
    Flat R M,
    ∀ (I : Ideal R), Function.Injective ⇑(rTensor M (Submodule.subtype I)),
    ∀ {ι : Type u} [Fintype ι] {f : ι → R} {x : ι → M}, ∑ i, f i ⊗ₜ x i = (0 : R ⊗[R] M) →
      VanishesTrivially R f x,
    ∀ {ι : Type u} [Fintype ι] {f : ι → R} {x : ι → M}, ∑ i, f i • x i = 0 → IsTrivialRelation f x,
    ∀ {ι : Type u} [Fintype ι] {f : ι →₀ R} {x : (ι →₀ R) →ₗ[R] M}, x f = 0 →
      ∃ (κ : Type u) (_ : Fintype κ) (a : (ι →₀ R) →ₗ[R] (κ →₀ R)) (y : (κ →₀ R) →ₗ[R] M),
        x = y ∘ₗ a ∧ a f = 0,
    ∀ {N : Type u} [AddCommGroup N] [Module R N] [Free R N] [Module.Finite R N] {f : N}
      {x : N →ₗ[R] M}, x f = 0 →
        ∃ (κ : Type u) (_ : Fintype κ) (a : N →ₗ[R] (κ →₀ R)) (y : (κ →₀ R) →ₗ[R] M),
          x = y ∘ₗ a ∧ a f = 0] := by
  classical
  tfae_have 1 ↔ 2 := iff_rTensor_injective' R M
  tfae_have 3 ↔ 2 := forall_vanishesTrivially_iff_forall_rTensor_injective R
  tfae_have 3 ↔ 4 := by
    simp [(TensorProduct.lid R M).injective.eq_iff.symm, isTrivialRelation_iff_vanishesTrivially]
  tfae_have 4 → 5
  | h₄, ι, hι, f, x, hfx => by
    let f' : ι → R := f
    let x' : ι → M := fun i ↦ x (single i 1)
    have := calc
      ∑ i, f' i • x' i
      _ = ∑ i, f i • x (single i 1)         := rfl
      _ = x (∑ i, f i • Finsupp.single i 1) := by simp_rw [map_sum, map_smul]
      _ = x f                               := by
        simp_rw [smul_single, smul_eq_mul, mul_one, univ_sum_single]
      _ = 0                                 := hfx
    obtain ⟨κ, hκ, a', y', ⟨ha'y', ha'⟩⟩ := h₄ this
    use κ, hκ
    use Finsupp.linearCombination R (fun i ↦ equivFunOnFinite.symm (a' i))
    use Finsupp.linearCombination R y'
    constructor
    · apply Finsupp.basisSingleOne.ext
      intro i
      simpa [linearCombination_apply, sum_fintype, Finsupp.single_apply] using ha'y' i
    · ext j
      simp only [linearCombination_apply, zero_smul, implies_true, sum_fintype, finset_sum_apply]
      exact ha' j
  tfae_have 5 → 4
  | h₅, ι, hi, f, x, hfx => by
    let f' : ι →₀ R := equivFunOnFinite.symm f
    let x' : (ι →₀ R) →ₗ[R] M := Finsupp.linearCombination R x
    have : x' f' = 0 := by simpa [x', f', linearCombination_apply, sum_fintype] using hfx
    obtain ⟨κ, hκ, a', y', ha'y', ha'⟩ := h₅ this
    refine ⟨κ, hκ, fun i ↦ a' (single i 1), fun j ↦ y' (single j 1), fun i ↦ ?_, fun j ↦ ?_⟩
    · simpa [x', ← map_smul, ← map_sum, smul_single] using
        LinearMap.congr_fun ha'y' (Finsupp.single i 1)
    · simp_rw [← smul_eq_mul, ← Finsupp.smul_apply, ← map_smul, ← finset_sum_apply, ← map_sum,
        smul_single, smul_eq_mul, mul_one,
        ← (fun _ ↦ equivFunOnFinite_symm_apply_toFun _ _ : ∀ x, f' x = f x), univ_sum_single]
      simpa using DFunLike.congr_fun ha' j
  tfae_have 5 → 6
  | h₅, N, _, _, _, _, f, x, hfx => by
    have ϕ := Module.Free.repr R N
    have : (x ∘ₗ ϕ.symm) (ϕ f) = 0 := by simpa
    obtain ⟨κ, hκ, a', y, ha'y, ha'⟩ := h₅ this
    refine ⟨κ, hκ, a' ∘ₗ ϕ, y, ?_, ?_⟩
    · simpa [LinearMap.comp_assoc] using congrArg (fun g ↦ (g ∘ₗ ϕ : N →ₗ[R] M)) ha'y
    · simpa using ha'
  tfae_have 6 → 5
  | h₆, _, _, _, _, hfx => h₆ hfx
  tfae_finish


/-- **Equational criterion for flatness** [Stacks 00HK](https://stacks.math.columbia.edu/tag/00HK).

A module $M$ is flat if and only if every relation $\sum_i f_i x_i = 0$ in $M$ is trivial. -/
theorem iff_forall_isTrivialRelation : Flat R M ↔ ∀ {ι : Type u} [Fintype ι] {f : ι → R}
    {x : ι → M}, ∑ i, f i • x i = 0 → IsTrivialRelation f x :=
  /-
    R M : Type u
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Eq ((List.cons (Module.Flat R M) (List.cons (∀ (I : Ideal R), Function.Injec …
  -/
  /-
    🎉 no goals
  -/
  (tfae_equational_criterion R M).out 0 3
  /-
    🎉 no goals
  -/


/-- **Equational criterion for flatness**
[Stacks 00HK](https://stacks.math.columbia.edu/tag/00HK), forward direction.

If $M$ is flat, then every relation $\sum_i f_i x_i = 0$ in $M$ is trivial. -/
theorem isTrivialRelation_of_sum_smul_eq_zero [Flat R M] {ι : Type u} [Fintype ι] {f : ι → R}
    {x : ι → M} (h : ∑ i, f i • x i = 0) : IsTrivialRelation f x :=
  iff_forall_isTrivialRelation.mp ‹Flat R M› h


/-- **Equational criterion for flatness**
[Stacks 00HK](https://stacks.math.columbia.edu/tag/00HK), backward direction.

If every relation $\sum_i f_i x_i = 0$ in $M$ is trivial, then $M$ is flat. -/
theorem of_forall_isTrivialRelation (hfx : ∀ {ι : Type u} [Fintype ι] {f : ι → R} {x : ι → M},
    ∑ i, f i • x i = 0 → IsTrivialRelation f x) : Flat R M :=
  iff_forall_isTrivialRelation.mpr hfx


/-- **Equational criterion for flatness**
[Stacks 00HK](https://stacks.math.columbia.edu/tag/00HK), alternate form.

A module $M$ is flat if and only if for all finite free modules $R^\iota$,
all $f \in R^{\iota}$, and all homomorphisms $x \colon R^{\iota} \to M$ such that $x(f) = 0$, there
exist a finite free module $R^\kappa$ and homomorphisms $a \colon R^{\iota} \to R^{\kappa}$ and
$y \colon R^{\kappa} \to M$ such that $x = y \circ a$ and $a(f) = 0$. -/
theorem iff_forall_exists_factorization : Flat R M ↔
    ∀ {ι : Type u} [Fintype ι] {f : ι →₀ R} {x : (ι →₀ R) →ₗ[R] M}, x f = 0 →
      ∃ (κ : Type u) (_ : Fintype κ) (a : (ι →₀ R) →ₗ[R] (κ →₀ R)) (y : (κ →₀ R) →ₗ[R] M),
                                /-
                                  R M : Type u
                                  inst✝² : CommRing R
                                  inst✝¹ : AddCommGroup M
                                  inst✝ : Module R M
                                  ⊢ Eq ((List.cons (Module.Flat R M) (List.cons (∀ (I : Ideal R), Function.Injec …
                                -/
                                /-
                                  🎉 no goals
                                -/
        x = y ∘ₗ a ∧ a f = 0 := (tfae_equational_criterion R M).out 0 4
                                /-
                                  🎉 no goals
                                -/


/-- **Equational criterion for flatness**
[Stacks 00HK](https://stacks.math.columbia.edu/tag/00HK), forward direction, alternate form.

Let $M$ be a flat module. Let $R^\iota$ be a finite free module, let $f \in R^{\iota}$ be an
element, and let $x \colon R^{\iota} \to M$ be a homomorphism such that $x(f) = 0$. Then there
exist a finite free module $R^\kappa$ and homomorphisms $a \colon R^{\iota} \to R^{\kappa}$ and
$y \colon R^{\kappa} \to M$ such that $x = y \circ a$ and $a(f) = 0$. -/
theorem exists_factorization_of_apply_eq_zero [Flat R M] {ι : Type u} [_root_.Finite ι]
    {f : ι →₀ R} {x : (ι →₀ R) →ₗ[R] M} (h : x f = 0) :
    ∃ (κ : Type u) (_ : Fintype κ) (a : (ι →₀ R) →ₗ[R] (κ →₀ R)) (y : (κ →₀ R) →ₗ[R] M),
      x = y ∘ₗ a ∧ a f = 0 :=
  let ⟨_⟩ := nonempty_fintype ι; iff_forall_exists_factorization.mp ‹Flat R M› h


/-- **Equational criterion for flatness**
[Stacks 00HK](https://stacks.math.columbia.edu/tag/00HK), backward direction, alternate form.

Let $M$ be a module over a commutative ring $R$. Suppose that for all finite free modules $R^\iota$,
all $f \in R^{\iota}$, and all homomorphisms $x \colon R^{\iota} \to M$ such that $x(f) = 0$, there
exist a finite free module $R^\kappa$ and homomorphisms $a \colon R^{\iota} \to R^{\kappa}$ and
$y \colon R^{\kappa} \to M$ such that $x = y \circ a$ and $a(f) = 0$. Then $M$ is flat. -/
theorem of_forall_exists_factorization (h : ∀ {ι : Type u} [Fintype ι] {f : ι →₀ R}
    {x : (ι →₀ R) →ₗ[R] M}, x f = 0 →
      ∃ (κ : Type u) (_ : Fintype κ) (a : (ι →₀ R) →ₗ[R] (κ →₀ R)) (y : (κ →₀ R) →ₗ[R] M),
      x = y ∘ₗ a ∧ a f = 0) : Flat R M := iff_forall_exists_factorization.mpr h


/-- **Equational criterion for flatness** [Stacks 00HK](https://stacks.math.columbia.edu/tag/00HK),
forward direction, second alternate form.

Let $M$ be a flat module over a commutative ring $R$. Let $N$ be a finite free module over $R$,
let $f \in N$, and let $x \colon N \to M$ be a homomorphism such that $x(f) = 0$. Then there exist a
finite index type $\kappa$ and module homomorphisms $a \colon N \to R^{\kappa}$ and
$y \colon R^{\kappa} \to M$ such that $x = y \circ a$ and $a(f) = 0$. -/
theorem exists_factorization_of_apply_eq_zero_of_free [Flat R M] {N : Type u} [AddCommGroup N]
    [Module R N] [Free R N] [Module.Finite R N] {f : N} {x : N →ₗ[R] M} (h : x f = 0) :
    ∃ (κ : Type u) (_ : Fintype κ) (a : N →ₗ[R] (κ →₀ R)) (y : (κ →₀ R) →ₗ[R] M),
      x = y ∘ₗ a ∧ a f = 0 := by
  /-
    R M : Type u
    inst✝⁷ : CommRing R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Module R M
    inst✝⁴ : Module.Flat R M
    N : Type u
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : Module.Free R N
    inst✝ : Module.Finite R N
    f : N
    x : LinearMap (RingHom.id R) N M
    h : Eq (x f) 0
    ⊢ Exists fun κ => Exists fun x_1 => Exists fun a => Exists fun y => And (Eq x  …
  -/
  exact ((tfae_equational_criterion R M).out 0 5 rfl rfl).mp ‹Flat R M› h
  /-
    🎉 no goals
  -/


/-- Let $M$ be a flat module. Let $K$ and $N$ be finite $R$-modules with $N$
free, and let $f \colon K \to N$ and $x \colon N \to M$ be homomorphisms such that
$x \circ f = 0$. Then there exist a finite index type $\kappa$ and module homomorphisms
$a \colon N \to R^{\kappa}$ and $y \colon R^{\kappa} \to M$ such that $x = y \circ a$ and
$a \circ f = 0$. -/
theorem exists_factorization_of_comp_eq_zero_of_free [Flat R M] {K N : Type u} [AddCommGroup K]
    [Module R K] [Module.Finite R K] [AddCommGroup N] [Module R N] [Free R N] [Module.Finite R N]
    {f : K →ₗ[R] N} {x : N →ₗ[R] M} (h : x ∘ₗ f = 0) :
    ∃ (κ : Type u) (_ : Fintype κ) (a : N →ₗ[R] (κ →₀ R)) (y : (κ →₀ R) →ₗ[R] M),
      x = y ∘ₗ a ∧ a ∘ₗ f = 0 := by
  have (K' : Submodule R K) (hK' : K'.FG) : ∃ (κ : Type u) (_ : Fintype κ) (a : N →ₗ[R] (κ →₀ R))
      (y : (κ →₀ R) →ₗ[R] M), x = y ∘ₗ a ∧ K' ≤ LinearMap.ker (a ∘ₗ f) := by
    revert N
    apply Submodule.fg_induction (N := K') (hN := hK')
    · intro k N _ _ _ _ f x hfx
      have : x (f k) = 0 := by simpa using LinearMap.congr_fun hfx k
      simpa using exists_factorization_of_apply_eq_zero_of_free this
    · intro K₁ K₂ ih₁ ih₂ N _ _ _ _ f x hfx
      obtain ⟨κ₁, _, a₁, y₁, rfl, ha₁⟩ := ih₁ hfx
      have : y₁ ∘ₗ (a₁ ∘ₗ f) = 0 := by rw [← comp_assoc, hfx]
      obtain ⟨κ₂, hκ₂, a₂, y₂, rfl, ha₂⟩ := ih₂ this
      use κ₂, hκ₂, a₂ ∘ₗ a₁, y₂
      simp_rw [comp_assoc]
      exact ⟨trivial, sup_le (ha₁.trans (ker_le_ker_comp _ _)) ha₂⟩
  /-
    R M : Type u
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    inst✝⁷ : Module.Flat R M
    K N : Type u
    inst✝⁶ : AddCommGroup K
    inst✝⁵ : Module R K
    inst✝⁴ : Module.Finite R K
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : Module.Free R N
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) K N
    x : LinearMap (RingHom.id R) N M
    h : Eq (x.comp f) 0
    this : ∀ (K' : Submodule R K), K'.FG → Exists fun κ => Exists fun x_1 => Exist …
    ⊢ Exists fun κ => Exists fun x_1 => Exists fun a => Exists fun y => And (Eq x  …
  -/
  convert this ⊤ Finite.out
  /-
    case h.e'_2.h.h.e'_2.h.h.e'_2.h.h.e'_2.h.h.e'_2.a
    R M : Type u
    inst✝¹⁰ : CommRing R
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    inst✝⁷ : Module.Flat R M
    K N : Type u
    inst✝⁶ : AddCommGroup K
    inst✝⁵ : Module R K
    inst✝⁴ : Module.Finite R K
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : Module.Free R N
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) K N
    x : LinearMap (RingHom.id R) N M
    h : Eq (x.comp f) 0
    this : ∀ (K' : Submodule R K), K'.FG → Exists fun κ => Exists fun x_1 => Exist …
    x✝³ : Type u
    x✝² : Fintype x✝³
    x✝¹ : LinearMap (RingHom.id R) N (Finsupp x✝³ R)
    x✝ : LinearMap (RingHom.id R) (Finsupp x✝³ R) M
    ⊢ Iff (Eq (x✝¹.comp f) 0) (LE.le Top.top (LinearMap.ker (x✝¹.comp f)))
  -/
  simp only [top_le_iff, ker_eq_top]
  /-
    🎉 no goals
  -/


/-- Every homomorphism from a finitely presented module to a flat module factors through a finite
free module. -/
theorem exists_factorization_of_isFinitelyPresented [Flat R M] {P : Type u} [AddCommGroup P]
    [Module R P] [FinitePresentation R P] (h₁ : P →ₗ[R] M) :
      ∃ (κ : Type u) (_ : Fintype κ) (h₂ : P →ₗ[R] (κ →₀ R)) (h₃ : (κ →₀ R) →ₗ[R] M),
        h₁ = h₃ ∘ₗ h₂ := by
  /-
    R M : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Flat R M
    P : Type u
    inst✝² : AddCommGroup P
    inst✝¹ : Module R P
    inst✝ : Module.FinitePresentation R P
    h₁ : LinearMap (RingHom.id R) P M
    ⊢ Exists fun κ => Exists fun x => Exists fun h₂ => Exists fun h₃ => Eq h₁ (h₃. …
  -/
  obtain ⟨L, _, _, K, ϕ, _, _, hK⟩ := FinitePresentation.equiv_quotient R P
  /-
    case intro.intro.intro.intro.intro.intro.intro
    R M : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Flat R M
    P : Type u
    inst✝² : AddCommGroup P
    inst✝¹ : Module R P
    inst✝ : Module.FinitePresentation R P
    h₁ : LinearMap (RingHom.id R) P M
    L : Type u
    w✝¹ : AddCommGroup L
    w✝ : Module R L
    K : Submodule R L
    ϕ : LinearEquiv (RingHom.id R) P (HasQuotient.Quotient L K)
    left✝¹ : Module.Free R L
    left✝ : Module.Finite R L
    hK : K.FG
    ⊢ Exists fun κ => Exists fun x => Exists fun h₂ => Exists fun h₃ => Eq h₁ (h₃. …
  -/
  haveI : Module.Finite R ↥K := Module.Finite.iff_fg.mpr hK
  have : (h₁ ∘ₗ ϕ.symm ∘ₗ K.mkQ) ∘ₗ K.subtype = 0 := by
    simp_rw [comp_assoc, (LinearMap.exact_subtype_mkQ K).linearMap_comp_eq_zero, comp_zero]
  /-
    case intro.intro.intro.intro.intro.intro.intro
    R M : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Flat R M
    P : Type u
    inst✝² : AddCommGroup P
    inst✝¹ : Module R P
    inst✝ : Module.FinitePresentation R P
    h₁ : LinearMap (RingHom.id R) P M
    L : Type u
    w✝¹ : AddCommGroup L
    w✝ : Module R L
    K : Submodule R L
    ϕ : LinearEquiv (RingHom.id R) P (HasQuotient.Quotient L K)
    left✝¹ : Module.Free R L
    left✝ : Module.Finite R L
    hK : K.FG
    this✝ : Module.Finite R (Subtype fun x => Membership.mem K x)
    this : Eq ((h₁.comp ((↑ϕ.symm).comp K.mkQ)).comp K.subtype) 0
    ⊢ Exists fun κ => Exists fun x => Exists fun h₂ => Exists fun h₃ => Eq h₁ (h₃. …
  -/
  obtain ⟨κ, hκ, a, y, hay, ha⟩ := exists_factorization_of_comp_eq_zero_of_free this
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    R M : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Flat R M
    P : Type u
    inst✝² : AddCommGroup P
    inst✝¹ : Module R P
    inst✝ : Module.FinitePresentation R P
    h₁ : LinearMap (RingHom.id R) P M
    L : Type u
    w✝¹ : AddCommGroup L
    w✝ : Module R L
    K : Submodule R L
    ϕ : LinearEquiv (RingHom.id R) P (HasQuotient.Quotient L K)
    left✝¹ : Module.Free R L
    left✝ : Module.Finite R L
    hK : K.FG
    this✝ : Module.Finite R (Subtype fun x => Membership.mem K x)
    this : Eq ((h₁.comp ((↑ϕ.symm).comp K.mkQ)).comp K.subtype) 0
    κ : Type u
    hκ : Fintype κ
    a : LinearMap (RingHom.id R) L (Finsupp κ R)
    y : LinearMap (RingHom.id R) (Finsupp κ R) M
    hay : Eq (h₁.comp ((↑ϕ.symm).comp K.mkQ)) (y.comp a)
    ha : Eq (a.comp K.subtype) 0
    ⊢ Exists fun κ => Exists fun x => Exists fun h₂ => Exists fun h₃ => Eq h₁ (h₃. …
  -/
  use κ, hκ, (K.liftQ a (by rwa [← range_le_ker_iff, Submodule.range_subtype] at ha)) ∘ₗ ϕ, y
  /-
    case h
    R M : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Flat R M
    P : Type u
    inst✝² : AddCommGroup P
    inst✝¹ : Module R P
    inst✝ : Module.FinitePresentation R P
    h₁ : LinearMap (RingHom.id R) P M
    L : Type u
    w✝¹ : AddCommGroup L
    w✝ : Module R L
    K : Submodule R L
    ϕ : LinearEquiv (RingHom.id R) P (HasQuotient.Quotient L K)
    left✝¹ : Module.Free R L
    left✝ : Module.Finite R L
    hK : K.FG
    this✝ : Module.Finite R (Subtype fun x => Membership.mem K x)
    this : Eq ((h₁.comp ((↑ϕ.symm).comp K.mkQ)).comp K.subtype) 0
    κ : Type u
    hκ : Fintype κ
    a : LinearMap (RingHom.id R) L (Finsupp κ R)
    y : LinearMap (RingHom.id R) (Finsupp κ R) M
    hay : Eq (h₁.comp ((↑ϕ.symm).comp K.mkQ)) (y.comp a)
    ha : Eq (a.comp K.subtype) 0
    ⊢ Eq h₁ (y.comp ((K.liftQ a ⋯).comp ↑ϕ))
  -/
  apply (cancel_right ϕ.symm.surjective).mp
  /-
    case h
    R M : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Flat R M
    P : Type u
    inst✝² : AddCommGroup P
    inst✝¹ : Module R P
    inst✝ : Module.FinitePresentation R P
    h₁ : LinearMap (RingHom.id R) P M
    L : Type u
    w✝¹ : AddCommGroup L
    w✝ : Module R L
    K : Submodule R L
    ϕ : LinearEquiv (RingHom.id R) P (HasQuotient.Quotient L K)
    left✝¹ : Module.Free R L
    left✝ : Module.Finite R L
    hK : K.FG
    this✝ : Module.Finite R (Subtype fun x => Membership.mem K x)
    this : Eq ((h₁.comp ((↑ϕ.symm).comp K.mkQ)).comp K.subtype) 0
    κ : Type u
    hκ : Fintype κ
    a : LinearMap (RingHom.id R) L (Finsupp κ R)
    y : LinearMap (RingHom.id R) (Finsupp κ R) M
    hay : Eq (h₁.comp ((↑ϕ.symm).comp K.mkQ)) (y.comp a)
    ha : Eq (a.comp K.subtype) 0
    ⊢ Eq (h₁.comp ↑ϕ.symm) ((y.comp ((K.liftQ a ⋯).comp ↑ϕ)).comp ↑ϕ.symm)
  -/
  apply (cancel_right K.mkQ_surjective).mp
  /-
    case h
    R M : Type u
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Flat R M
    P : Type u
    inst✝² : AddCommGroup P
    inst✝¹ : Module R P
    inst✝ : Module.FinitePresentation R P
    h₁ : LinearMap (RingHom.id R) P M
    L : Type u
    w✝¹ : AddCommGroup L
    w✝ : Module R L
    K : Submodule R L
    ϕ : LinearEquiv (RingHom.id R) P (HasQuotient.Quotient L K)
    left✝¹ : Module.Free R L
    left✝ : Module.Finite R L
    hK : K.FG
    this✝ : Module.Finite R (Subtype fun x => Membership.mem K x)
    this : Eq ((h₁.comp ((↑ϕ.symm).comp K.mkQ)).comp K.subtype) 0
    κ : Type u
    hκ : Fintype κ
    a : LinearMap (RingHom.id R) L (Finsupp κ R)
    y : LinearMap (RingHom.id R) (Finsupp κ R) M
    hay : Eq (h₁.comp ((↑ϕ.symm).comp K.mkQ)) (y.comp a)
    ha : Eq (a.comp K.subtype) 0
    ⊢ Eq ((h₁.comp ↑ϕ.symm).comp K.mkQ) (((y.comp ((K.liftQ a ⋯).comp ↑ϕ)).comp ↑ϕ …
  -/
  simpa [comp_assoc]
  /-
    🎉 no goals
  -/



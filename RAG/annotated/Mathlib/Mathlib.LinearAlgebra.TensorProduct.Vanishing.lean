variable (m n) in
/-- An expression $\sum_i m_i \otimes n_i$ in $M \otimes N$
*vanishes trivially* if there exist a finite index type $\kappa$,
elements $(y_j)_{j \in \kappa}$ of $N$, and elements $(a_{ij})_{i \in \iota, j \in \kappa}$ of $R$
such that for all $i$,
$$n_i = \sum_j a_{ij} y_j$$
and for all $j$,
$$\sum_{i} a_{ij} m_i = 0.$$
Note that this condition is not symmetric in $M$ and $N$.
(The terminology "trivial" comes from [Stacks 00HK](https://stacks.math.columbia.edu/tag/00HK).)-/
abbrev VanishesTrivially : Prop :=
  ∃ (κ : Type u) (_ : Fintype κ) (a : ι → κ → R) (y : κ → N),
    (∀ i, n i = ∑ j, a i j • y j) ∧ ∀ j, ∑ i, a i j • m i = 0


/-- **Equational criterion for vanishing**
[A. Altman and S. Kleiman, *A term of commutative algebra* (Lemma 8.16)][altman2021term],
backward direction.

If the expression $\sum_i m_i \otimes n_i$ vanishes trivially, then it vanishes.
That is, $\sum_i m_i \otimes n_i = 0$. -/
theorem sum_tmul_eq_zero_of_vanishesTrivially (hmn : VanishesTrivially R m n) :
    ∑ i, m i ⊗ₜ n i = (0 : M ⊗[R] N) := by
  /-
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hmn : TensorProduct.VanishesTrivially R m n
    ⊢ Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
  -/
  obtain ⟨κ, _, a, y, h₁, h₂⟩ := hmn
  /-
    case intro.intro.intro.intro.intro
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    κ : Type u
    w✝ : Fintype κ
    a : ι → κ → R
    y : κ → N
    h₁ : ∀ (i : ι), Eq (n i) (Finset.univ.sum fun j => HSMul.hSMul (a i j) (y j))
    h₂ : ∀ (j : κ), Eq (Finset.univ.sum fun i => HSMul.hSMul (a i j) (m i)) 0
    ⊢ Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
  -/
  simp_rw [h₁, tmul_sum, tmul_smul]
  /-
    case intro.intro.intro.intro.intro
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    κ : Type u
    w✝ : Fintype κ
    a : ι → κ → R
    y : κ → N
    h₁ : ∀ (i : ι), Eq (n i) (Finset.univ.sum fun j => HSMul.hSMul (a i j) (y j))
    h₂ : ∀ (j : κ), Eq (Finset.univ.sum fun i => HSMul.hSMul (a i j) (m i)) 0
    ⊢ Eq (Finset.univ.sum fun x => Finset.univ.sum fun x_1 => HSMul.hSMul (a x x_1 …
  -/
  rw [Finset.sum_comm]
  /-
    case intro.intro.intro.intro.intro
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    κ : Type u
    w✝ : Fintype κ
    a : ι → κ → R
    y : κ → N
    h₁ : ∀ (i : ι), Eq (n i) (Finset.univ.sum fun j => HSMul.hSMul (a i j) (y j))
    h₂ : ∀ (j : κ), Eq (Finset.univ.sum fun i => HSMul.hSMul (a i j) (m i)) 0
    ⊢ Eq (Finset.univ.sum fun y_1 => Finset.univ.sum fun x => HSMul.hSMul (a x y_1 …
  -/
  simp_rw [← tmul_smul, ← smul_tmul, ← sum_tmul, h₂, zero_tmul, Finset.sum_const_zero]
  /-
    🎉 no goals
  -/


/-- **Equational criterion for vanishing**
[A. Altman and S. Kleiman, *A term of commutative algebra* (Lemma 8.16)][altman2021term],
forward direction.

Assume that the $m_i$ generate $M$. If the expression $\sum_i m_i \otimes n_i$
vanishes, then it vanishes trivially. -/
theorem vanishesTrivially_of_sum_tmul_eq_zero (hm : Submodule.span R (Set.range m) = ⊤)
    (hmn : ∑ i, m i ⊗ₜ n i = (0 : M ⊗[R] N)) : VanishesTrivially R m n := by
  -- Define a map $G \colon R^\iota \to M$ whose matrix entries are the $m_i$. It is surjective.
  /-
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hm : Eq (Submodule.span R (Set.range m)) Top.top
    hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
    ⊢ TensorProduct.VanishesTrivially R m n
  -/
  set G : (ι →₀ R) →ₗ[R] M := Finsupp.linearCombination R m with hG
  /-
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hm : Eq (Submodule.span R (Set.range m)) Top.top
    hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
    G : LinearMap (RingHom.id R) (Finsupp ι R) M := Finsupp.linearCombination R m
    hG : Eq G (Finsupp.linearCombination R m)
    ⊢ TensorProduct.VanishesTrivially R m n
  -/
  have G_basis_eq (i : ι) : G (Finsupp.single i 1) = m i := by simp [hG, toModule_lof]
  have G_surjective : Surjective G := by
    apply LinearMap.range_eq_top.mp
    apply top_le_iff.mp
    rw [← hm]
    apply Submodule.span_le.mpr
    rintro _ ⟨i, rfl⟩
    use Finsupp.single i 1, G_basis_eq i
  /- Consider the element $\sum_i e_i \otimes n_i$ of $R^\iota \otimes N$. It is in the kernel of
  $R^\iota \otimes N \to M \otimes N$. -/
  /-
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hm : Eq (Submodule.span R (Set.range m)) Top.top
    hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
    G : LinearMap (RingHom.id R) (Finsupp ι R) M := Finsupp.linearCombination R m
    hG : Eq G (Finsupp.linearCombination R m)
    G_basis_eq : ∀ (i : ι), Eq (G (Finsupp.single i 1)) (m i)
    G_surjective : Function.Surjective ⇑G
    ⊢ TensorProduct.VanishesTrivially R m n
  -/
  set en : (ι →₀ R) ⊗[R] N := ∑ i, Finsupp.single i 1 ⊗ₜ n i with hen
  /-
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hm : Eq (Submodule.span R (Set.range m)) Top.top
    hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
    G : LinearMap (RingHom.id R) (Finsupp ι R) M := Finsupp.linearCombination R m
    hG : Eq G (Finsupp.linearCombination R m)
    G_basis_eq : ∀ (i : ι), Eq (G (Finsupp.single i 1)) (m i)
    G_surjective : Function.Surjective ⇑G
    en : TensorProduct R (Finsupp ι R) N := Finset.univ.sum fun i => TensorProduct …
    hen : Eq en (Finset.univ.sum fun i => TensorProduct.tmul R (Finsupp.single i 1 …
    ⊢ TensorProduct.VanishesTrivially R m n
  -/
  have en_mem_ker : en ∈ ker (rTensor N G) := by simp [hen, G_basis_eq, hmn]
  -- We have an exact sequence $\ker G \to R^\iota \to M \to 0$.
  /-
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hm : Eq (Submodule.span R (Set.range m)) Top.top
    hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
    G : LinearMap (RingHom.id R) (Finsupp ι R) M := Finsupp.linearCombination R m
    hG : Eq G (Finsupp.linearCombination R m)
    G_basis_eq : ∀ (i : ι), Eq (G (Finsupp.single i 1)) (m i)
    G_surjective : Function.Surjective ⇑G
    en : TensorProduct R (Finsupp ι R) N := Finset.univ.sum fun i => TensorProduct …
    hen : Eq en (Finset.univ.sum fun i => TensorProduct.tmul R (Finsupp.single i 1 …
    en_mem_ker : Membership.mem (LinearMap.ker (LinearMap.rTensor N G)) en
    ⊢ TensorProduct.VanishesTrivially R m n
  -/
  have exact_ker_subtype : Exact (ker G).subtype G := G.exact_subtype_ker_map
  -- Tensor the exact sequence with $N$.
  have exact_rTensor_ker_subtype : Exact (rTensor N (ker G).subtype) (rTensor N G) :=
    rTensor_exact (M := ↥(ker G)) N exact_ker_subtype G_surjective
  /- We conclude that $\sum_i e_i \otimes n_i$ is in the range of
    $\ker G \otimes N \to R^\iota \otimes N$. -/
  have en_mem_range : en ∈ range (rTensor N (ker G).subtype) :=
    exact_rTensor_ker_subtype.linearMap_ker_eq ▸ en_mem_ker
  /- There is an element of in $\ker G \otimes N$ that maps to $\sum_i e_i \otimes n_i$.
  Write it as a finite sum of pure tensors. -/
  /-
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hm : Eq (Submodule.span R (Set.range m)) Top.top
    hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
    G : LinearMap (RingHom.id R) (Finsupp ι R) M := Finsupp.linearCombination R m
    hG : Eq G (Finsupp.linearCombination R m)
    G_basis_eq : ∀ (i : ι), Eq (G (Finsupp.single i 1)) (m i)
    G_surjective : Function.Surjective ⇑G
    en : TensorProduct R (Finsupp ι R) N := Finset.univ.sum fun i => TensorProduct …
    hen : Eq en (Finset.univ.sum fun i => TensorProduct.tmul R (Finsupp.single i 1 …
    en_mem_ker : Membership.mem (LinearMap.ker (LinearMap.rTensor N G)) en
    exact_ker_subtype : Function.Exact ⇑(LinearMap.ker G).subtype ⇑G
    exact_rTensor_ker_subtype : Function.Exact ⇑(LinearMap.rTensor N (LinearMap.ke …
    en_mem_range : Membership.mem (LinearMap.range (LinearMap.rTensor N (LinearMap …
    ⊢ TensorProduct.VanishesTrivially R m n
  -/
  obtain ⟨kn, hkn⟩ := en_mem_range
  /-
    case intro
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hm : Eq (Submodule.span R (Set.range m)) Top.top
    hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
    G : LinearMap (RingHom.id R) (Finsupp ι R) M := Finsupp.linearCombination R m
    hG : Eq G (Finsupp.linearCombination R m)
    G_basis_eq : ∀ (i : ι), Eq (G (Finsupp.single i 1)) (m i)
    G_surjective : Function.Surjective ⇑G
    en : TensorProduct R (Finsupp ι R) N := Finset.univ.sum fun i => TensorProduct …
    hen : Eq en (Finset.univ.sum fun i => TensorProduct.tmul R (Finsupp.single i 1 …
    en_mem_ker : Membership.mem (LinearMap.ker (LinearMap.rTensor N G)) en
    exact_ker_subtype : Function.Exact ⇑(LinearMap.ker G).subtype ⇑G
    exact_rTensor_ker_subtype : Function.Exact ⇑(LinearMap.rTensor N (LinearMap.ke …
    kn : TensorProduct R (Subtype fun x => Membership.mem (LinearMap.ker G) x) N
    hkn : Eq ((LinearMap.rTensor N (LinearMap.ker G).subtype) kn) en
    ⊢ TensorProduct.VanishesTrivially R m n
  -/
  obtain ⟨ma, rfl : kn = ∑ kj ∈ ma, kj.1 ⊗ₜ[R] kj.2⟩ := exists_finset kn
  /-
    case intro.intro
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hm : Eq (Submodule.span R (Set.range m)) Top.top
    hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
    G : LinearMap (RingHom.id R) (Finsupp ι R) M := Finsupp.linearCombination R m
    hG : Eq G (Finsupp.linearCombination R m)
    G_basis_eq : ∀ (i : ι), Eq (G (Finsupp.single i 1)) (m i)
    G_surjective : Function.Surjective ⇑G
    en : TensorProduct R (Finsupp ι R) N := Finset.univ.sum fun i => TensorProduct …
    hen : Eq en (Finset.univ.sum fun i => TensorProduct.tmul R (Finsupp.single i 1 …
    en_mem_ker : Membership.mem (LinearMap.ker (LinearMap.rTensor N G)) en
    exact_ker_subtype : Function.Exact ⇑(LinearMap.ker G).subtype ⇑G
    exact_rTensor_ker_subtype : Function.Exact ⇑(LinearMap.rTensor N (LinearMap.ke …
    ma : Finset (Prod (Subtype fun x => Membership.mem (LinearMap.ker G) x) N)
    hkn : Eq ((LinearMap.rTensor N (LinearMap.ker G).subtype) (ma.sum fun kj => Te …
    ⊢ TensorProduct.VanishesTrivially R m n
  -/
  use ↑↑ma, FinsetCoe.fintype ma
  /- Let $\sum_j k_j \otimes y_j$ be the sum obtained in the previous step.
  In order to show that $\sum_i m_i \otimes n_i$ vanishes trivially, it suffices to prove that there
  exist $(a_{ij})_{i, j}$ such that for all $i$,
  $$n_i = \sum_j a_{ij} y_j$$
  and for all $j$,
  $$\sum_{i} a_{ij} m_i = 0.$$
  For this, take $a_{ij}$ to be the coefficient of $e_i$ in $k_j$. -/
  /-
    case h
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hm : Eq (Submodule.span R (Set.range m)) Top.top
    hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
    G : LinearMap (RingHom.id R) (Finsupp ι R) M := Finsupp.linearCombination R m
    hG : Eq G (Finsupp.linearCombination R m)
    G_basis_eq : ∀ (i : ι), Eq (G (Finsupp.single i 1)) (m i)
    G_surjective : Function.Surjective ⇑G
    en : TensorProduct R (Finsupp ι R) N := Finset.univ.sum fun i => TensorProduct …
    hen : Eq en (Finset.univ.sum fun i => TensorProduct.tmul R (Finsupp.single i 1 …
    en_mem_ker : Membership.mem (LinearMap.ker (LinearMap.rTensor N G)) en
    exact_ker_subtype : Function.Exact ⇑(LinearMap.ker G).subtype ⇑G
    exact_rTensor_ker_subtype : Function.Exact ⇑(LinearMap.rTensor N (LinearMap.ke …
    ma : Finset (Prod (Subtype fun x => Membership.mem (LinearMap.ker G) x) N)
    hkn : Eq ((LinearMap.rTensor N (LinearMap.ker G).subtype) (ma.sum fun kj => Te …
    ⊢ Exists fun a => Exists fun y => And (∀ (i : ι), Eq (n i) (Finset.univ.sum fu …
  -/
  use fun i ⟨⟨kj, _⟩, _⟩ ↦ (kj : ι →₀ R) i
  /-
    case h
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hm : Eq (Submodule.span R (Set.range m)) Top.top
    hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
    G : LinearMap (RingHom.id R) (Finsupp ι R) M := Finsupp.linearCombination R m
    hG : Eq G (Finsupp.linearCombination R m)
    G_basis_eq : ∀ (i : ι), Eq (G (Finsupp.single i 1)) (m i)
    G_surjective : Function.Surjective ⇑G
    en : TensorProduct R (Finsupp ι R) N := Finset.univ.sum fun i => TensorProduct …
    hen : Eq en (Finset.univ.sum fun i => TensorProduct.tmul R (Finsupp.single i 1 …
    en_mem_ker : Membership.mem (LinearMap.ker (LinearMap.rTensor N G)) en
    exact_ker_subtype : Function.Exact ⇑(LinearMap.ker G).subtype ⇑G
    exact_rTensor_ker_subtype : Function.Exact ⇑(LinearMap.rTensor N (LinearMap.ke …
    ma : Finset (Prod (Subtype fun x => Membership.mem (LinearMap.ker G) x) N)
    hkn : Eq ((LinearMap.rTensor N (LinearMap.ker G).subtype) (ma.sum fun kj => Te …
    ⊢ Exists fun y => And (∀ (i : ι), Eq (n i) (Finset.univ.sum fun j => HSMul.hSM …
  -/
  use fun ⟨⟨_, yj⟩, _⟩ ↦ yj
  /-
    case h
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hm : Eq (Submodule.span R (Set.range m)) Top.top
    hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
    G : LinearMap (RingHom.id R) (Finsupp ι R) M := Finsupp.linearCombination R m
    hG : Eq G (Finsupp.linearCombination R m)
    G_basis_eq : ∀ (i : ι), Eq (G (Finsupp.single i 1)) (m i)
    G_surjective : Function.Surjective ⇑G
    en : TensorProduct R (Finsupp ι R) N := Finset.univ.sum fun i => TensorProduct …
    hen : Eq en (Finset.univ.sum fun i => TensorProduct.tmul R (Finsupp.single i 1 …
    en_mem_ker : Membership.mem (LinearMap.ker (LinearMap.rTensor N G)) en
    exact_ker_subtype : Function.Exact ⇑(LinearMap.ker G).subtype ⇑G
    exact_rTensor_ker_subtype : Function.Exact ⇑(LinearMap.rTensor N (LinearMap.ke …
    ma : Finset (Prod (Subtype fun x => Membership.mem (LinearMap.ker G) x) N)
    hkn : Eq ((LinearMap.rTensor N (LinearMap.ker G).subtype) (ma.sum fun kj => Te …
    ⊢ And (∀ (i : ι), Eq (n i) (Finset.univ.sum fun j => HSMul.hSMul (TensorProduc …
  -/
  constructor
    /-
      case h.left
      R : Type u
      inst✝⁵ : CommRing R
      M : Type u
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      N : Type u
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      ι : Type u
      inst✝ : Fintype ι
      m : ι → M
      n : ι → N
      hm : Eq (Submodule.span R (Set.range m)) Top.top
      hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
      G : LinearMap (RingHom.id R) (Finsupp ι R) M := Finsupp.linearCombination R m
      hG : Eq G (Finsupp.linearCombination R m)
      G_basis_eq : ∀ (i : ι), Eq (G (Finsupp.single i 1)) (m i)
      G_surjective : Function.Surjective ⇑G
      en : TensorProduct R (Finsupp ι R) N := Finset.univ.sum fun i => TensorProduct …
      hen : Eq en (Finset.univ.sum fun i => TensorProduct.tmul R (Finsupp.single i 1 …
      en_mem_ker : Membership.mem (LinearMap.ker (LinearMap.rTensor N G)) en
      exact_ker_subtype : Function.Exact ⇑(LinearMap.ker G).subtype ⇑G
      exact_rTensor_ker_subtype : Function.Exact ⇑(LinearMap.rTensor N (LinearMap.ke …
      ma : Finset (Prod (Subtype fun x => Membership.mem (LinearMap.ker G) x) N)
      hkn : Eq ((LinearMap.rTensor N (LinearMap.ker G).subtype) (ma.sum fun kj => Te …
      ⊢ ∀ (i : ι), Eq (n i) (Finset.univ.sum fun j => HSMul.hSMul (TensorProduct.van …
    -/
  · intro i
    classical
    apply_fun finsuppScalarLeft R N ι at hkn
    apply_fun (· i) at hkn
    symm at hkn
    simp only [map_sum, finsuppScalarLeft_apply_tmul, zero_smul, Finsupp.single_zero,
      Finsupp.sum_single_index, one_smul, Finsupp.finset_sum_apply, Finsupp.single_apply,
      Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte, rTensor_tmul, coe_subtype, Finsupp.sum_apply,
      Finsupp.sum_ite_eq', Finsupp.mem_support_iff, ne_eq, ite_not, en] at hkn
    simp only [Finset.univ_eq_attach, Finset.sum_attach ma (fun x ↦ (x.1 : ι →₀ R) i • x.2)]
    convert hkn using 2 with x _
    split
    · next h'x => rw [h'x, zero_smul]
    · rfl
    /-
      case h.right
      R : Type u
      inst✝⁵ : CommRing R
      M : Type u
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      N : Type u
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      ι : Type u
      inst✝ : Fintype ι
      m : ι → M
      n : ι → N
      hm : Eq (Submodule.span R (Set.range m)) Top.top
      hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
      G : LinearMap (RingHom.id R) (Finsupp ι R) M := Finsupp.linearCombination R m
      hG : Eq G (Finsupp.linearCombination R m)
      G_basis_eq : ∀ (i : ι), Eq (G (Finsupp.single i 1)) (m i)
      G_surjective : Function.Surjective ⇑G
      en : TensorProduct R (Finsupp ι R) N := Finset.univ.sum fun i => TensorProduct …
      hen : Eq en (Finset.univ.sum fun i => TensorProduct.tmul R (Finsupp.single i 1 …
      en_mem_ker : Membership.mem (LinearMap.ker (LinearMap.rTensor N G)) en
      exact_ker_subtype : Function.Exact ⇑(LinearMap.ker G).subtype ⇑G
      exact_rTensor_ker_subtype : Function.Exact ⇑(LinearMap.rTensor N (LinearMap.ke …
      ma : Finset (Prod (Subtype fun x => Membership.mem (LinearMap.ker G) x) N)
      hkn : Eq ((LinearMap.rTensor N (LinearMap.ker G).subtype) (ma.sum fun kj => Te …
      ⊢ ∀ (j : Subtype fun x => Membership.mem ma x), Eq (Finset.univ.sum fun i => H …
    -/
  · rintro ⟨⟨⟨k, hk⟩, _⟩, _⟩
    simpa only [hG, linearCombination_apply, zero_smul, implies_true, Finsupp.sum_fintype] using
      mem_ker.mp hk


/-- **Equational criterion for vanishing**
[A. Altman and S. Kleiman, *A term of commutative algebra* (Lemma 8.16)][altman2021term].

Assume that the $m_i$ generate $M$. Then the expression $\sum_i m_i \otimes n_i$ vanishes
trivially if and only if it vanishes. -/
theorem vanishesTrivially_iff_sum_tmul_eq_zero (hm : Submodule.span R (Set.range m) = ⊤) :
    VanishesTrivially R m n ↔ ∑ i, m i ⊗ₜ n i = (0 : M ⊗[R] N) :=
  ⟨sum_tmul_eq_zero_of_vanishesTrivially R, vanishesTrivially_of_sum_tmul_eq_zero R hm⟩


/-- **Equational criterion for vanishing**
[A. Altman and S. Kleiman, *A term of commutative algebra* (Lemma 8.16)][altman2021term],
forward direction, generalization.

Assume that the submodule $M' \subseteq M$ generated by the $m_i$
satisfies the property that the map $M' \otimes N \to M \otimes N$ is injective. If the expression
$\sum_i m_i \otimes n_i$ vanishes, then it vanishes trivially. -/
theorem vanishesTrivially_of_sum_tmul_eq_zero_of_rTensor_injective
    (hm : Injective (rTensor N (span R (Set.range m)).subtype))
    (hmn : ∑ i, m i ⊗ₜ n i = (0 : M ⊗[R] N)) : VanishesTrivially R m n := by
  -- Restrict `m` on the codomain to $M'$, then apply `vanishesTrivially_of_sum_tmul_eq_zero`.
  /-
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hm : Function.Injective ⇑(LinearMap.rTensor N (Submodule.span R (Set.range m)) …
    hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
    ⊢ TensorProduct.VanishesTrivially R m n
  -/
  have mem_M' i : m i ∈ span R (Set.range m) := subset_span ⟨i, rfl⟩
  /-
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hm : Function.Injective ⇑(LinearMap.rTensor N (Submodule.span R (Set.range m)) …
    hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
    mem_M' : ∀ (i : ι), Membership.mem (Submodule.span R (Set.range m)) (m i)
    ⊢ TensorProduct.VanishesTrivially R m n
  -/
  set m' : ι → span R (Set.range m) := Subtype.coind m mem_M' with m'_eq
  have hm' : span R (Set.range m') = ⊤ := by
    apply map_injective_of_injective (injective_subtype (span R (Set.range m)))
    rw [Submodule.map_span, Submodule.map_top, range_subtype, coe_subtype, ← Set.range_comp]
    rfl
  have hm'n : ∑ i, m' i ⊗ₜ n i = (0 : span R (Set.range m) ⊗[R] N) := by
    apply hm
    simp only [m'_eq, map_sum, rTensor_tmul, coe_subtype, Subtype.coind_coe, _root_.map_zero, hmn]
  /-
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hm : Function.Injective ⇑(LinearMap.rTensor N (Submodule.span R (Set.range m)) …
    hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
    mem_M' : ∀ (i : ι), Membership.mem (Submodule.span R (Set.range m)) (m i)
    m' : ι → Subtype fun x => Membership.mem (Submodule.span R (Set.range m)) x := …
    m'_eq : Eq m' (Subtype.coind m mem_M')
    hm' : Eq (Submodule.span R (Set.range m')) Top.top
    hm'n : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m' i) (n i)) 0
    ⊢ TensorProduct.VanishesTrivially R m n
  -/
  have : VanishesTrivially R m' n := vanishesTrivially_of_sum_tmul_eq_zero R hm' hm'n
  /-
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hm : Function.Injective ⇑(LinearMap.rTensor N (Submodule.span R (Set.range m)) …
    hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
    mem_M' : ∀ (i : ι), Membership.mem (Submodule.span R (Set.range m)) (m i)
    m' : ι → Subtype fun x => Membership.mem (Submodule.span R (Set.range m)) x := …
    m'_eq : Eq m' (Subtype.coind m mem_M')
    hm' : Eq (Submodule.span R (Set.range m')) Top.top
    hm'n : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m' i) (n i)) 0
    this : TensorProduct.VanishesTrivially R m' n
    ⊢ TensorProduct.VanishesTrivially R m n
  -/
  unfold VanishesTrivially at this ⊢
  /-
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hm : Function.Injective ⇑(LinearMap.rTensor N (Submodule.span R (Set.range m)) …
    hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
    mem_M' : ∀ (i : ι), Membership.mem (Submodule.span R (Set.range m)) (m i)
    m' : ι → Subtype fun x => Membership.mem (Submodule.span R (Set.range m)) x := …
    m'_eq : Eq m' (Subtype.coind m mem_M')
    hm' : Eq (Submodule.span R (Set.range m')) Top.top
    hm'n : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m' i) (n i)) 0
    this : Exists fun κ => Exists fun x => Exists fun a => Exists fun y => And (∀  …
    ⊢ Exists fun κ => Exists fun x => Exists fun a => Exists fun y => And (∀ (i :  …
  -/
  convert this with κ _ a y j
  /-
    case h.e'_2.h.h.e'_2.h.h.e'_2.h.h.e'_2.h.h.e'_2.h.a
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hm : Function.Injective ⇑(LinearMap.rTensor N (Submodule.span R (Set.range m)) …
    hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
    mem_M' : ∀ (i : ι), Membership.mem (Submodule.span R (Set.range m)) (m i)
    m' : ι → Subtype fun x => Membership.mem (Submodule.span R (Set.range m)) x := …
    m'_eq : Eq m' (Subtype.coind m mem_M')
    hm' : Eq (Submodule.span R (Set.range m')) Top.top
    hm'n : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m' i) (n i)) 0
    this : Exists fun κ => Exists fun x => Exists fun a => Exists fun y => And (∀  …
    κ : Type u
    x✝ : Fintype κ
    a : ι → κ → R
    y : κ → N
    j : κ
    ⊢ Iff (Eq (Finset.univ.sum fun i => HSMul.hSMul (a i j) (m i)) 0) (Eq (Finset. …
  -/
  convert (injective_iff_map_eq_zero' _).mp (injective_subtype (span R (Set.range m))) _
  /-
    case h.e'_1.h.e'_2
    R : Type u
    inst✝⁵ : CommRing R
    M : Type u
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    N : Type u
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    ι : Type u
    inst✝ : Fintype ι
    m : ι → M
    n : ι → N
    hm : Function.Injective ⇑(LinearMap.rTensor N (Submodule.span R (Set.range m)) …
    hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
    mem_M' : ∀ (i : ι), Membership.mem (Submodule.span R (Set.range m)) (m i)
    m' : ι → Subtype fun x => Membership.mem (Submodule.span R (Set.range m)) x := …
    m'_eq : Eq m' (Subtype.coind m mem_M')
    hm' : Eq (Submodule.span R (Set.range m')) Top.top
    hm'n : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m' i) (n i)) 0
    this : Exists fun κ => Exists fun x => Exists fun a => Exists fun y => And (∀  …
    κ : Type u
    x✝ : Fintype κ
    a : ι → κ → R
    y : κ → N
    j : κ
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (a i j) (m i)) ((Submodule.span R ( …
  -/
  simp [m'_eq]
  /-
    🎉 no goals
  -/


/-- **Equational criterion for vanishing**
[A. Altman and S. Kleiman, *A term of commutative algebra* (Lemma 8.16)][altman2021term],
generalization.

Assume that the submodule $M' \subseteq M$ generated by the $m_i$ satisfies the
property that the map $M' \otimes N \to M \otimes N$ is injective. Then the expression
$\sum_i m_i \otimes n_i$ vanishes trivially if and only if it vanishes. -/
theorem vanishesTrivially_iff_sum_tmul_eq_zero_of_rTensor_injective
    (hm : Injective (rTensor N (span R (Set.range m)).subtype)) :
    VanishesTrivially R m n ↔ ∑ i, m i ⊗ₜ n i = (0 : M ⊗[R] N) :=
  ⟨sum_tmul_eq_zero_of_vanishesTrivially R,
    vanishesTrivially_of_sum_tmul_eq_zero_of_rTensor_injective R hm⟩


/-- Converse of `TensorProduct.vanishesTrivially_of_sum_tmul_eq_zero_of_rTensor_injective`.

Assume that every expression $\sum_i m_i \otimes n_i$ which vanishes also vanishes trivially.
Then, for every submodule $M' \subseteq M$, the map $M' \otimes N \to M \otimes N$ is injective. -/
theorem rTensor_injective_of_forall_vanishesTrivially
    (hMN : ∀ {ι : Type u} [Fintype ι] {m : ι → M} {n : ι → N},
      ∑ i, m i ⊗ₜ n i = (0 : M ⊗[R] N) → VanishesTrivially R m n)
    (M' : Submodule R M) : Injective (rTensor N M'.subtype) := by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    hMN : ∀ {ι : Type u} [inst : Fintype ι] {m : ι → M} {n : ι → N}, Eq (Finset.un …
    M' : Submodule R M
    ⊢ Function.Injective ⇑(LinearMap.rTensor N M'.subtype)
  -/
  apply (injective_iff_map_eq_zero _).mpr
  /-
    R : Type u
    inst✝⁴ : CommRing R
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    hMN : ∀ {ι : Type u} [inst : Fintype ι] {m : ι → M} {n : ι → N}, Eq (Finset.un …
    M' : Submodule R M
    ⊢ ∀ (a : TensorProduct R (Subtype fun x => Membership.mem M' x) N), Eq ((Linea …
  -/
  rintro x hx
  /-
    R : Type u
    inst✝⁴ : CommRing R
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    hMN : ∀ {ι : Type u} [inst : Fintype ι] {m : ι → M} {n : ι → N}, Eq (Finset.un …
    M' : Submodule R M
    x : TensorProduct R (Subtype fun x => Membership.mem M' x) N
    hx : Eq ((LinearMap.rTensor N M'.subtype) x) 0
    ⊢ Eq x 0
  -/
  obtain ⟨s, rfl⟩ := exists_finset x
  /-
    case intro
    R : Type u
    inst✝⁴ : CommRing R
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    hMN : ∀ {ι : Type u} [inst : Fintype ι] {m : ι → M} {n : ι → N}, Eq (Finset.un …
    M' : Submodule R M
    s : Finset (Prod (Subtype fun x => Membership.mem M' x) N)
    hx : Eq ((LinearMap.rTensor N M'.subtype) (s.sum fun i => TensorProduct.tmul R …
    ⊢ Eq (s.sum fun i => TensorProduct.tmul R i.1 i.2) 0
  -/
  rw [← Finset.sum_attach]
  /-
    case intro
    R : Type u
    inst✝⁴ : CommRing R
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    hMN : ∀ {ι : Type u} [inst : Fintype ι] {m : ι → M} {n : ι → N}, Eq (Finset.un …
    M' : Submodule R M
    s : Finset (Prod (Subtype fun x => Membership.mem M' x) N)
    hx : Eq ((LinearMap.rTensor N M'.subtype) (s.sum fun i => TensorProduct.tmul R …
    ⊢ Eq (s.attach.sum fun x => TensorProduct.tmul R (↑x).1 (↑x).2) 0
  -/
  apply sum_tmul_eq_zero_of_vanishesTrivially
  /-
    case intro.hmn
    R : Type u
    inst✝⁴ : CommRing R
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    hMN : ∀ {ι : Type u} [inst : Fintype ι] {m : ι → M} {n : ι → N}, Eq (Finset.un …
    M' : Submodule R M
    s : Finset (Prod (Subtype fun x => Membership.mem M' x) N)
    hx : Eq ((LinearMap.rTensor N M'.subtype) (s.sum fun i => TensorProduct.tmul R …
    ⊢ TensorProduct.VanishesTrivially R (fun i => (↑i).1) fun i => (↑i).2
  -/
  simp only [map_sum, rTensor_tmul, coe_subtype] at hx
  /-
    case intro.hmn
    R : Type u
    inst✝⁴ : CommRing R
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    hMN : ∀ {ι : Type u} [inst : Fintype ι] {m : ι → M} {n : ι → N}, Eq (Finset.un …
    M' : Submodule R M
    s : Finset (Prod (Subtype fun x => Membership.mem M' x) N)
    hx : Eq (s.sum fun x => TensorProduct.tmul R (↑x.1) x.2) 0
    ⊢ TensorProduct.VanishesTrivially R (fun i => (↑i).1) fun i => (↑i).2
  -/
  have := hMN ((Finset.sum_attach s _).trans hx)
  /-
    case intro.hmn
    R : Type u
    inst✝⁴ : CommRing R
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    hMN : ∀ {ι : Type u} [inst : Fintype ι] {m : ι → M} {n : ι → N}, Eq (Finset.un …
    M' : Submodule R M
    s : Finset (Prod (Subtype fun x => Membership.mem M' x) N)
    hx : Eq (s.sum fun x => TensorProduct.tmul R (↑x.1) x.2) 0
    this : TensorProduct.VanishesTrivially R (fun i => ↑(↑i).1) fun i => (↑i).2
    ⊢ TensorProduct.VanishesTrivially R (fun i => (↑i).1) fun i => (↑i).2
  -/
  unfold VanishesTrivially at this ⊢
  /-
    case intro.hmn
    R : Type u
    inst✝⁴ : CommRing R
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    hMN : ∀ {ι : Type u} [inst : Fintype ι] {m : ι → M} {n : ι → N}, Eq (Finset.un …
    M' : Submodule R M
    s : Finset (Prod (Subtype fun x => Membership.mem M' x) N)
    hx : Eq (s.sum fun x => TensorProduct.tmul R (↑x.1) x.2) 0
    this : Exists fun κ => Exists fun x => Exists fun a => Exists fun y => And (∀  …
    ⊢ Exists fun κ => Exists fun x => Exists fun a => Exists fun y => And (∀ (i :  …
  -/
  convert this with κ _ a y j
  /-
    case h.e'_2.h.h.e'_2.h.h.e'_2.h.h.e'_2.h.h.e'_2.h.a
    R : Type u
    inst✝⁴ : CommRing R
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    hMN : ∀ {ι : Type u} [inst : Fintype ι] {m : ι → M} {n : ι → N}, Eq (Finset.un …
    M' : Submodule R M
    s : Finset (Prod (Subtype fun x => Membership.mem M' x) N)
    hx : Eq (s.sum fun x => TensorProduct.tmul R (↑x.1) x.2) 0
    this : Exists fun κ => Exists fun x => Exists fun a => Exists fun y => And (∀  …
    κ : Type u
    x✝ : Fintype κ
    a : (Subtype fun x => Membership.mem s x) → κ → R
    y : κ → N
    j : κ
    ⊢ Iff (Eq (Finset.univ.sum fun i => HSMul.hSMul (a i j) ((fun i => (↑i).1) i)) …
  -/
  symm
  /-
    case h.e'_2.h.h.e'_2.h.h.e'_2.h.h.e'_2.h.h.e'_2.h.a
    R : Type u
    inst✝⁴ : CommRing R
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    hMN : ∀ {ι : Type u} [inst : Fintype ι] {m : ι → M} {n : ι → N}, Eq (Finset.un …
    M' : Submodule R M
    s : Finset (Prod (Subtype fun x => Membership.mem M' x) N)
    hx : Eq (s.sum fun x => TensorProduct.tmul R (↑x.1) x.2) 0
    this : Exists fun κ => Exists fun x => Exists fun a => Exists fun y => And (∀  …
    κ : Type u
    x✝ : Fintype κ
    a : (Subtype fun x => Membership.mem s x) → κ → R
    y : κ → N
    j : κ
    ⊢ Iff (Eq (Finset.univ.sum fun i => HSMul.hSMul (a i j) ((fun i => ↑(↑i).1) i) …
  -/
  convert (injective_iff_map_eq_zero' _).mp (injective_subtype M') _
  /-
    case h.e'_1.h.e'_2
    R : Type u
    inst✝⁴ : CommRing R
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    hMN : ∀ {ι : Type u} [inst : Fintype ι] {m : ι → M} {n : ι → N}, Eq (Finset.un …
    M' : Submodule R M
    s : Finset (Prod (Subtype fun x => Membership.mem M' x) N)
    hx : Eq (s.sum fun x => TensorProduct.tmul R (↑x.1) x.2) 0
    this : Exists fun κ => Exists fun x => Exists fun a => Exists fun y => And (∀  …
    κ : Type u
    x✝ : Fintype κ
    a : (Subtype fun x => Membership.mem s x) → κ → R
    y : κ → N
    j : κ
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (a i j) ((fun i => ↑(↑i).1) i)) (M' …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Every expression $\sum_i m_i \otimes n_i$ which vanishes also vanishes trivially if and only if
for every submodule $M' \subseteq M$, the map $M' \otimes N \to M \otimes N$ is injective. -/
theorem forall_vanishesTrivially_iff_forall_rTensor_injective :
    (∀ {ι : Type u} [Fintype ι] {m : ι → M} {n : ι → N},
      ∑ i, m i ⊗ₜ n i = (0 : M ⊗[R] N) → VanishesTrivially R m n) ↔
    ∀ M' : Submodule R M, Injective (rTensor N M'.subtype) := by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    ⊢ Iff (∀ {ι : Type u} [inst : Fintype ι] {m : ι → M} {n : ι → N}, Eq (Finset.u …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝⁴ : CommRing R
      M : Type u
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      ⊢ (∀ {ι : Type u} [inst : Fintype ι] {m : ι → M} {n : ι → N}, Eq (Finset.univ. …
    -/
  · intro h
    /-
      case mp
      R : Type u
      inst✝⁴ : CommRing R
      M : Type u
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      h : ∀ {ι : Type u} [inst : Fintype ι] {m : ι → M} {n : ι → N}, Eq (Finset.univ …
      ⊢ ∀ (M' : Submodule R M), Function.Injective ⇑(LinearMap.rTensor N M'.subtype)
    -/
    exact rTensor_injective_of_forall_vanishesTrivially R h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝⁴ : CommRing R
      M : Type u
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      ⊢ (∀ (M' : Submodule R M), Function.Injective ⇑(LinearMap.rTensor N M'.subtype …
    -/
  · intro h ι _ m n hmn
    /-
      case mpr
      R : Type u
      inst✝⁵ : CommRing R
      M : Type u
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      N : Type u
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      h : ∀ (M' : Submodule R M), Function.Injective ⇑(LinearMap.rTensor N M'.subtype)
      ι : Type u
      inst✝ : Fintype ι
      m : ι → M
      n : ι → N
      hmn : Eq (Finset.univ.sum fun i => TensorProduct.tmul R (m i) (n i)) 0
      ⊢ TensorProduct.VanishesTrivially R m n
    -/
    exact vanishesTrivially_of_sum_tmul_eq_zero_of_rTensor_injective R (h _) hmn
    /-
      🎉 no goals
    -/


/-- Every expression $\sum_i m_i \otimes n_i$ which vanishes also vanishes trivially if and only if
for every finitely generated submodule $M' \subseteq M$, the map $M' \otimes N \to M \otimes N$ is
injective. -/
theorem forall_vanishesTrivially_iff_forall_FG_rTensor_injective :
    (∀ {ι : Type u} [Fintype ι] {m : ι → M} {n : ι → N},
      ∑ i, m i ⊗ₜ n i = (0 : M ⊗[R] N) → VanishesTrivially R m n) ↔
    ∀ (M' : Submodule R M) (_ : M'.FG), Injective (rTensor N M'.subtype) := by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    M : Type u
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type u
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    ⊢ Iff (∀ {ι : Type u} [inst : Fintype ι] {m : ι → M} {n : ι → N}, Eq (Finset.u …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝⁴ : CommRing R
      M : Type u
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      ⊢ (∀ {ι : Type u} [inst : Fintype ι] {m : ι → M} {n : ι → N}, Eq (Finset.univ. …
    -/
  · intro h M' _
    /-
      case mp
      R : Type u
      inst✝⁴ : CommRing R
      M : Type u
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      h : ∀ {ι : Type u} [inst : Fintype ι] {m : ι → M} {n : ι → N}, Eq (Finset.univ …
      M' : Submodule R M
      x✝ : M'.FG
      ⊢ Function.Injective ⇑(LinearMap.rTensor N M'.subtype)
    -/
    exact rTensor_injective_of_forall_vanishesTrivially R h M'
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝⁴ : CommRing R
      M : Type u
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type u
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      ⊢ (∀ (M' : Submodule R M), M'.FG → Function.Injective ⇑(LinearMap.rTensor N M' …
    -/
  · intro h ι _ m n hmn
    exact vanishesTrivially_of_sum_tmul_eq_zero_of_rTensor_injective R
      (h _ (fg_span (Set.finite_range _))) hmn


/-- If the map $M' \otimes N \to M \otimes N$ is injective for every finitely generated submodule
$M' \subseteq M$, then it is in fact injective for every submodule $M' \subseteq M$. -/
theorem rTensor_injective_of_forall_FG_rTensor_injective
    (hMN : ∀ (M' : Submodule R M) (_ : M'.FG), Injective (rTensor N M'.subtype))
    (M' : Submodule R M) : Injective (rTensor N M'.subtype) :=
  (forall_vanishesTrivially_iff_forall_rTensor_injective R).mp
    ((forall_vanishesTrivially_iff_forall_FG_rTensor_injective R).mpr hMN) M'



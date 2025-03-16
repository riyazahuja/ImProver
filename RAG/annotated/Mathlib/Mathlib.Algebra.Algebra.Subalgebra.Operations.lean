theorem ker_rangeRestrict (f : A →ₐ[R] B) : RingHom.ker f.rangeRestrict = RingHom.ker f :=
  Ideal.ext fun _ ↦ Subtype.ext_iff


/-- Suppose we are given `∑ i, lᵢ * sᵢ = 1` ∈ `S`, and `S'` a subalgebra of `S` that contains
`lᵢ` and `sᵢ`. To check that an `x : S` falls in `S'`, we only need to show that
`sᵢ ^ n • x ∈ S'` for some `n` for each `sᵢ`. -/
theorem mem_of_finset_sum_eq_one_of_pow_smul_mem
    {ι : Type*} (ι' : Finset ι) (s : ι → S) (l : ι → S)
    (e : ∑ i ∈ ι', l i * s i = 1) (hs : ∀ i, s i ∈ S') (hl : ∀ i, l i ∈ S') (x : S)
    (H : ∀ i, ∃ n : ℕ, (s i ^ n : S) • x ∈ S') : x ∈ S' := by
  -- Porting note: needed to add this instance
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    S' : Subalgebra R S
    ι : Type u_3
    ι' : Finset ι
    s l : ι → S
    e : Eq (ι'.sum fun i => HMul.hMul (l i) (s i)) 1
    hs : ∀ (i : ι), Membership.mem S' (s i)
    hl : ∀ (i : ι), Membership.mem S' (l i)
    x : S
    H : ∀ (i : ι), Exists fun n => Membership.mem S' (HSMul.hSMul (HPow.hPow (s i) …
    ⊢ Membership.mem S' x
  -/
  let _i : Algebra { x // x ∈ S' } { x // x ∈ S' } := Algebra.id _
  suffices x ∈ Subalgebra.toSubmodule (Algebra.ofId S' S).range by
    obtain ⟨x, rfl⟩ := this
    exact x.2
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    S' : Subalgebra R S
    ι : Type u_3
    ι' : Finset ι
    s l : ι → S
    e : Eq (ι'.sum fun i => HMul.hMul (l i) (s i)) 1
    hs : ∀ (i : ι), Membership.mem S' (s i)
    hl : ∀ (i : ι), Membership.mem S' (l i)
    x : S
    H : ∀ (i : ι), Exists fun n => Membership.mem S' (HSMul.hSMul (HPow.hPow (s i) …
    _i : Algebra (Subtype fun x => Membership.mem S' x) (Subtype fun x => Membersh …
    ⊢ Membership.mem (Subalgebra.toSubmodule (Algebra.ofId (Subtype fun x => Membe …
  -/
  choose n hn using H
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    S' : Subalgebra R S
    ι : Type u_3
    ι' : Finset ι
    s l : ι → S
    e : Eq (ι'.sum fun i => HMul.hMul (l i) (s i)) 1
    hs : ∀ (i : ι), Membership.mem S' (s i)
    hl : ∀ (i : ι), Membership.mem S' (l i)
    x : S
    _i : Algebra (Subtype fun x => Membership.mem S' x) (Subtype fun x => Membersh …
    n : ι → Nat
    hn : ∀ (i : ι), Membership.mem S' (HSMul.hSMul (HPow.hPow (s i) (n i)) x)
    ⊢ Membership.mem (Subalgebra.toSubmodule (Algebra.ofId (Subtype fun x => Membe …
  -/
  let s' : ι → S' := fun x => ⟨s x, hs x⟩
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    S' : Subalgebra R S
    ι : Type u_3
    ι' : Finset ι
    s l : ι → S
    e : Eq (ι'.sum fun i => HMul.hMul (l i) (s i)) 1
    hs : ∀ (i : ι), Membership.mem S' (s i)
    hl : ∀ (i : ι), Membership.mem S' (l i)
    x : S
    _i : Algebra (Subtype fun x => Membership.mem S' x) (Subtype fun x => Membersh …
    n : ι → Nat
    hn : ∀ (i : ι), Membership.mem S' (HSMul.hSMul (HPow.hPow (s i) (n i)) x)
    s' : ι → Subtype fun x => Membership.mem S' x := fun x => ⟨s x, ⋯⟩
    ⊢ Membership.mem (Subalgebra.toSubmodule (Algebra.ofId (Subtype fun x => Membe …
  -/
  let l' : ι → S' := fun x => ⟨l x, hl x⟩
  have e' : ∑ i ∈ ι', l' i * s' i = 1 := by
    ext
    show S'.subtype (∑ i ∈ ι', l' i * s' i) = 1
    simpa only [map_sum, map_mul] using e
  have : Ideal.span (s' '' ι') = ⊤ := by
    rw [Ideal.eq_top_iff_one, ← e']
    apply sum_mem
    intros i hi
    exact Ideal.mul_mem_left _ _ <| Ideal.subset_span <| Set.mem_image_of_mem s' hi
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    S' : Subalgebra R S
    ι : Type u_3
    ι' : Finset ι
    s l : ι → S
    e : Eq (ι'.sum fun i => HMul.hMul (l i) (s i)) 1
    hs : ∀ (i : ι), Membership.mem S' (s i)
    hl : ∀ (i : ι), Membership.mem S' (l i)
    x : S
    _i : Algebra (Subtype fun x => Membership.mem S' x) (Subtype fun x => Membersh …
    n : ι → Nat
    hn : ∀ (i : ι), Membership.mem S' (HSMul.hSMul (HPow.hPow (s i) (n i)) x)
    s' : ι → Subtype fun x => Membership.mem S' x := fun x => ⟨s x, ⋯⟩
    l' : ι → Subtype fun x => Membership.mem S' x := fun x => ⟨l x, ⋯⟩
    e' : Eq (ι'.sum fun i => HMul.hMul (l' i) (s' i)) 1
    this : Eq (Ideal.span (Set.image s' ↑ι')) Top.top
    ⊢ Membership.mem (Subalgebra.toSubmodule (Algebra.ofId (Subtype fun x => Membe …
  -/
  let N := ι'.sup n
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    S' : Subalgebra R S
    ι : Type u_3
    ι' : Finset ι
    s l : ι → S
    e : Eq (ι'.sum fun i => HMul.hMul (l i) (s i)) 1
    hs : ∀ (i : ι), Membership.mem S' (s i)
    hl : ∀ (i : ι), Membership.mem S' (l i)
    x : S
    _i : Algebra (Subtype fun x => Membership.mem S' x) (Subtype fun x => Membersh …
    n : ι → Nat
    hn : ∀ (i : ι), Membership.mem S' (HSMul.hSMul (HPow.hPow (s i) (n i)) x)
    s' : ι → Subtype fun x => Membership.mem S' x := fun x => ⟨s x, ⋯⟩
    l' : ι → Subtype fun x => Membership.mem S' x := fun x => ⟨l x, ⋯⟩
    e' : Eq (ι'.sum fun i => HMul.hMul (l' i) (s' i)) 1
    this : Eq (Ideal.span (Set.image s' ↑ι')) Top.top
    N : Nat := ι'.sup n
    ⊢ Membership.mem (Subalgebra.toSubmodule (Algebra.ofId (Subtype fun x => Membe …
  -/
  have hN := Ideal.span_pow_eq_top _ this N
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    S' : Subalgebra R S
    ι : Type u_3
    ι' : Finset ι
    s l : ι → S
    e : Eq (ι'.sum fun i => HMul.hMul (l i) (s i)) 1
    hs : ∀ (i : ι), Membership.mem S' (s i)
    hl : ∀ (i : ι), Membership.mem S' (l i)
    x : S
    _i : Algebra (Subtype fun x => Membership.mem S' x) (Subtype fun x => Membersh …
    n : ι → Nat
    hn : ∀ (i : ι), Membership.mem S' (HSMul.hSMul (HPow.hPow (s i) (n i)) x)
    s' : ι → Subtype fun x => Membership.mem S' x := fun x => ⟨s x, ⋯⟩
    l' : ι → Subtype fun x => Membership.mem S' x := fun x => ⟨l x, ⋯⟩
    e' : Eq (ι'.sum fun i => HMul.hMul (l' i) (s' i)) 1
    this : Eq (Ideal.span (Set.image s' ↑ι')) Top.top
    N : Nat := ι'.sup n
    hN : Eq (Ideal.span (Set.image (fun x => HPow.hPow x N) (Set.image s' ↑ι'))) T …
    ⊢ Membership.mem (Subalgebra.toSubmodule (Algebra.ofId (Subtype fun x => Membe …
  -/
  apply (Algebra.ofId S' S).range.toSubmodule.mem_of_span_top_of_smul_mem _ hN
  /-
    case H
    R : Type u_1
    S : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    S' : Subalgebra R S
    ι : Type u_3
    ι' : Finset ι
    s l : ι → S
    e : Eq (ι'.sum fun i => HMul.hMul (l i) (s i)) 1
    hs : ∀ (i : ι), Membership.mem S' (s i)
    hl : ∀ (i : ι), Membership.mem S' (l i)
    x : S
    _i : Algebra (Subtype fun x => Membership.mem S' x) (Subtype fun x => Membersh …
    n : ι → Nat
    hn : ∀ (i : ι), Membership.mem S' (HSMul.hSMul (HPow.hPow (s i) (n i)) x)
    s' : ι → Subtype fun x => Membership.mem S' x := fun x => ⟨s x, ⋯⟩
    l' : ι → Subtype fun x => Membership.mem S' x := fun x => ⟨l x, ⋯⟩
    e' : Eq (ι'.sum fun i => HMul.hMul (l' i) (s' i)) 1
    this : Eq (Ideal.span (Set.image s' ↑ι')) Top.top
    N : Nat := ι'.sup n
    hN : Eq (Ideal.span (Set.image (fun x => HPow.hPow x N) (Set.image s' ↑ι'))) T …
    ⊢ ∀ (r : ↑(Set.image (fun x => HPow.hPow x N) (Set.image s' ↑ι'))), Membership …
  -/
  rintro ⟨_, _, ⟨i, hi, rfl⟩, rfl⟩
  /-
    case H.mk.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    S' : Subalgebra R S
    ι : Type u_3
    ι' : Finset ι
    s l : ι → S
    e : Eq (ι'.sum fun i => HMul.hMul (l i) (s i)) 1
    hs : ∀ (i : ι), Membership.mem S' (s i)
    hl : ∀ (i : ι), Membership.mem S' (l i)
    x : S
    _i : Algebra (Subtype fun x => Membership.mem S' x) (Subtype fun x => Membersh …
    n : ι → Nat
    hn : ∀ (i : ι), Membership.mem S' (HSMul.hSMul (HPow.hPow (s i) (n i)) x)
    s' : ι → Subtype fun x => Membership.mem S' x := fun x => ⟨s x, ⋯⟩
    l' : ι → Subtype fun x => Membership.mem S' x := fun x => ⟨l x, ⋯⟩
    e' : Eq (ι'.sum fun i => HMul.hMul (l' i) (s' i)) 1
    this : Eq (Ideal.span (Set.image s' ↑ι')) Top.top
    N : Nat := ι'.sup n
    hN : Eq (Ideal.span (Set.image (fun x => HPow.hPow x N) (Set.image s' ↑ι'))) T …
    i : ι
    hi : Membership.mem (↑ι') i
    ⊢ Membership.mem (Subalgebra.toSubmodule (Algebra.ofId (Subtype fun x => Membe …
  -/
  change s' i ^ N • x ∈ _
  /-
    case H.mk.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    S' : Subalgebra R S
    ι : Type u_3
    ι' : Finset ι
    s l : ι → S
    e : Eq (ι'.sum fun i => HMul.hMul (l i) (s i)) 1
    hs : ∀ (i : ι), Membership.mem S' (s i)
    hl : ∀ (i : ι), Membership.mem S' (l i)
    x : S
    _i : Algebra (Subtype fun x => Membership.mem S' x) (Subtype fun x => Membersh …
    n : ι → Nat
    hn : ∀ (i : ι), Membership.mem S' (HSMul.hSMul (HPow.hPow (s i) (n i)) x)
    s' : ι → Subtype fun x => Membership.mem S' x := fun x => ⟨s x, ⋯⟩
    l' : ι → Subtype fun x => Membership.mem S' x := fun x => ⟨l x, ⋯⟩
    e' : Eq (ι'.sum fun i => HMul.hMul (l' i) (s' i)) 1
    this : Eq (Ideal.span (Set.image s' ↑ι')) Top.top
    N : Nat := ι'.sup n
    hN : Eq (Ideal.span (Set.image (fun x => HPow.hPow x N) (Set.image s' ↑ι'))) T …
    i : ι
    hi : Membership.mem (↑ι') i
    ⊢ Membership.mem (Subalgebra.toSubmodule (Algebra.ofId (Subtype fun x => Membe …
  -/
  rw [← tsub_add_cancel_of_le (show n i ≤ N from Finset.le_sup hi), pow_add, mul_smul]
  /-
    case H.mk.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    S' : Subalgebra R S
    ι : Type u_3
    ι' : Finset ι
    s l : ι → S
    e : Eq (ι'.sum fun i => HMul.hMul (l i) (s i)) 1
    hs : ∀ (i : ι), Membership.mem S' (s i)
    hl : ∀ (i : ι), Membership.mem S' (l i)
    x : S
    _i : Algebra (Subtype fun x => Membership.mem S' x) (Subtype fun x => Membersh …
    n : ι → Nat
    hn : ∀ (i : ι), Membership.mem S' (HSMul.hSMul (HPow.hPow (s i) (n i)) x)
    s' : ι → Subtype fun x => Membership.mem S' x := fun x => ⟨s x, ⋯⟩
    l' : ι → Subtype fun x => Membership.mem S' x := fun x => ⟨l x, ⋯⟩
    e' : Eq (ι'.sum fun i => HMul.hMul (l' i) (s' i)) 1
    this : Eq (Ideal.span (Set.image s' ↑ι')) Top.top
    N : Nat := ι'.sup n
    hN : Eq (Ideal.span (Set.image (fun x => HPow.hPow x N) (Set.image s' ↑ι'))) T …
    i : ι
    hi : Membership.mem (↑ι') i
    ⊢ Membership.mem (Subalgebra.toSubmodule (Algebra.ofId (Subtype fun x => Membe …
  -/
  refine Submodule.smul_mem _ (⟨_, pow_mem (hs i) _⟩ : S') ?_
  /-
    case H.mk.intro.intro.intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    S' : Subalgebra R S
    ι : Type u_3
    ι' : Finset ι
    s l : ι → S
    e : Eq (ι'.sum fun i => HMul.hMul (l i) (s i)) 1
    hs : ∀ (i : ι), Membership.mem S' (s i)
    hl : ∀ (i : ι), Membership.mem S' (l i)
    x : S
    _i : Algebra (Subtype fun x => Membership.mem S' x) (Subtype fun x => Membersh …
    n : ι → Nat
    hn : ∀ (i : ι), Membership.mem S' (HSMul.hSMul (HPow.hPow (s i) (n i)) x)
    s' : ι → Subtype fun x => Membership.mem S' x := fun x => ⟨s x, ⋯⟩
    l' : ι → Subtype fun x => Membership.mem S' x := fun x => ⟨l x, ⋯⟩
    e' : Eq (ι'.sum fun i => HMul.hMul (l' i) (s' i)) 1
    this : Eq (Ideal.span (Set.image s' ↑ι')) Top.top
    N : Nat := ι'.sup n
    hN : Eq (Ideal.span (Set.image (fun x => HPow.hPow x N) (Set.image s' ↑ι'))) T …
    i : ι
    hi : Membership.mem (↑ι') i
    ⊢ Membership.mem (Subalgebra.toSubmodule (Algebra.ofId (Subtype fun x => Membe …
  -/
  exact ⟨⟨_, hn i⟩, rfl⟩
  /-
    🎉 no goals
  -/


theorem mem_of_span_eq_top_of_smul_pow_mem
    (s : Set S) (l : s →₀ S) (hs : Finsupp.linearCombination S ((↑) : s → S) l = 1)
    (hs' : s ⊆ S') (hl : ∀ i, l i ∈ S') (x : S) (H : ∀ r : s, ∃ n : ℕ, (r : S) ^ n • x ∈ S') :
    x ∈ S' :=
  mem_of_finset_sum_eq_one_of_pow_smul_mem S' l.support (↑) l hs (fun x => hs' x.2) hl x H



/-- A ring `R` satisfies `IsLasker R` when any `I : Ideal R` can be decomposed into
finitely many primary ideals.-/
def IsLasker : Prop :=
  ∀ I : Ideal R, ∃ s : Finset (Ideal R), s.inf id = I ∧ ∀ ⦃J⦄, J ∈ s → J.IsPrimary


lemma decomposition_erase_inf [DecidableEq (Ideal R)] {I : Ideal R}
    {s : Finset (Ideal R)} (hs : s.inf id = I) :
    ∃ t : Finset (Ideal R), t ⊆ s ∧ t.inf id = I ∧ (∀ ⦃J⦄, J ∈ t → ¬ (t.erase J).inf id ≤ J) := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq (Ideal R)
    I : Ideal R
    s : Finset (Ideal R)
    hs : Eq (s.inf id) I
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And (Eq (t.inf id) I) (∀ ⦃J : Id …
  -/
  induction s using Finset.strongInductionOn
  /-
    case a
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq (Ideal R)
    I : Ideal R
    s✝ : Finset (Ideal R)
    a✝ : ∀ (t : Finset (Ideal R)), HasSSubset.SSubset t s✝ → Eq (t.inf id) I → Exi …
    hs : Eq (s✝.inf id) I
    ⊢ Exists fun t => And (HasSubset.Subset t s✝) (And (Eq (t.inf id) I) (∀ ⦃J : I …
  -/
  rename_i _ s IH
  /-
    case a
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq (Ideal R)
    I : Ideal R
    s : Finset (Ideal R)
    IH : ∀ (t : Finset (Ideal R)), HasSSubset.SSubset t s → Eq (t.inf id) I → Exis …
    hs : Eq (s.inf id) I
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And (Eq (t.inf id) I) (∀ ⦃J : Id …
  -/
  by_cases H : ∀ J ∈ s, ¬ (s.erase J).inf id ≤ J
    /-
      case pos
      R : Type u_1
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq (Ideal R)
      I : Ideal R
      s : Finset (Ideal R)
      IH : ∀ (t : Finset (Ideal R)), HasSSubset.SSubset t s → Eq (t.inf id) I → Exis …
      hs : Eq (s.inf id) I
      H : ∀ (J : Ideal R), Membership.mem s J → Not (LE.le ((s.erase J).inf id) J)
      ⊢ Exists fun t => And (HasSubset.Subset t s) (And (Eq (t.inf id) I) (∀ ⦃J : Id …
    -/
  · exact ⟨s, Finset.Subset.rfl, hs, H⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq (Ideal R)
    I : Ideal R
    s : Finset (Ideal R)
    IH : ∀ (t : Finset (Ideal R)), HasSSubset.SSubset t s → Eq (t.inf id) I → Exis …
    hs : Eq (s.inf id) I
    H : Not (∀ (J : Ideal R), Membership.mem s J → Not (LE.le ((s.erase J).inf id) …
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And (Eq (t.inf id) I) (∀ ⦃J : Id …
  -/
  push_neg at H
  /-
    case neg
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq (Ideal R)
    I : Ideal R
    s : Finset (Ideal R)
    IH : ∀ (t : Finset (Ideal R)), HasSSubset.SSubset t s → Eq (t.inf id) I → Exis …
    hs : Eq (s.inf id) I
    H : Exists fun J => And (Membership.mem s J) (LE.le ((s.erase J).inf id) J)
    ⊢ Exists fun t => And (HasSubset.Subset t s) (And (Eq (t.inf id) I) (∀ ⦃J : Id …
  -/
  obtain ⟨J, hJ, hJ'⟩ := H
  refine (IH (s.erase J) (Finset.erase_ssubset hJ) ?_).imp
    fun t ↦ And.imp_left (fun ht ↦ ht.trans (Finset.erase_subset _ _))
  /-
    case neg.intro.intro
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq (Ideal R)
    I : Ideal R
    s : Finset (Ideal R)
    IH : ∀ (t : Finset (Ideal R)), HasSSubset.SSubset t s → Eq (t.inf id) I → Exis …
    hs : Eq (s.inf id) I
    J : Ideal R
    hJ : Membership.mem s J
    hJ' : LE.le ((s.erase J).inf id) J
    ⊢ Eq ((s.erase J).inf id) I
  -/
  rw [← Finset.insert_erase hJ] at hs
  /-
    case neg.intro.intro
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq (Ideal R)
    I : Ideal R
    s : Finset (Ideal R)
    IH : ∀ (t : Finset (Ideal R)), HasSSubset.SSubset t s → Eq (t.inf id) I → Exis …
    J : Ideal R
    hs : Eq ((Insert.insert J (s.erase J)).inf id) I
    hJ : Membership.mem s J
    hJ' : LE.le ((s.erase J).inf id) J
    ⊢ Eq ((s.erase J).inf id) I
  -/
  simp [← hs, hJ']
  /-
    🎉 no goals
  -/


lemma isPrimary_decomposition_pairwise_ne_radical {I : Ideal R}
    {s : Finset (Ideal R)} (hs : s.inf id = I) (hs' : ∀ ⦃J⦄, J ∈ s → J.IsPrimary) :
    ∃ t : Finset (Ideal R), t.inf id = I ∧ (∀ ⦃J⦄, J ∈ t → J.IsPrimary) ∧
      (t : Set (Ideal R)).Pairwise ((· ≠ ·) on radical) := by
  classical
  refine ⟨(s.image (fun J ↦ s.filter (fun I ↦ I.radical = J.radical))).image fun t ↦ t.inf id,
    ?_, ?_, ?_⟩
  · rw [← hs]
    refine le_antisymm ?_ ?_ <;> intro x hx
    · simp only [Finset.inf_image, CompTriple.comp_eq, Submodule.mem_finset_inf,
      Function.comp_apply, Finset.mem_filter, id_eq, and_imp] at hx ⊢
      intro J hJ
      exact hx J hJ J hJ rfl
    · simp only [Submodule.mem_finset_inf, id_eq, Finset.inf_image, CompTriple.comp_eq,
      Function.comp_apply, Finset.mem_filter, and_imp] at hx ⊢
      intro J _ K hK _
      exact hx K hK
  · simp only [Finset.mem_image, exists_exists_and_eq_and, forall_exists_index, and_imp,
    forall_apply_eq_imp_iff₂]
    intro J hJ
    refine isPrimary_finset_inf (i := J) ?_ ?_ (by simp)
    · simp [hJ]
    · simp only [Finset.mem_filter, id_eq, and_imp]
      intro y hy
      simp [hs' hy]
  · intro I hI J hJ hIJ
    simp only [Finset.coe_image, Set.mem_image, Finset.mem_coe, exists_exists_and_eq_and] at hI hJ
    obtain ⟨I', hI', hI⟩ := hI
    obtain ⟨J', hJ', hJ⟩ := hJ
    simp only [Function.onFun, ne_eq]
    contrapose! hIJ
    suffices I'.radical = J'.radical by
      rw [← hI, ← hJ, this]
    · rw [← hI, radical_finset_inf (i := I') (by simp [hI']) (by simp), id_eq] at hIJ
      rw [hIJ, ← hJ, radical_finset_inf (i := J') (by simp [hJ']) (by simp), id_eq]


lemma exists_minimal_isPrimary_decomposition_of_isPrimary_decomposition [DecidableEq (Ideal R)]
    {I : Ideal R} {s : Finset (Ideal R)} (hs : s.inf id = I) (hs' : ∀ ⦃J⦄, J ∈ s → J.IsPrimary) :
    ∃ t : Finset (Ideal R), t.inf id = I ∧ (∀ ⦃J⦄, J ∈ t → J.IsPrimary) ∧
      ((t : Set (Ideal R)).Pairwise ((· ≠ ·) on radical)) ∧
      (∀ ⦃J⦄, J ∈ t → ¬ (t.erase J).inf id ≤ J) := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq (Ideal R)
    I : Ideal R
    s : Finset (Ideal R)
    hs : Eq (s.inf id) I
    hs' : ∀ ⦃J : Ideal R⦄, Membership.mem s J → J.IsPrimary
    ⊢ Exists fun t => And (Eq (t.inf id) I) (And (∀ ⦃J : Ideal R⦄, Membership.mem  …
  -/
  obtain ⟨t, ht, ht', ht''⟩ := isPrimary_decomposition_pairwise_ne_radical hs hs'
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq (Ideal R)
    I : Ideal R
    s : Finset (Ideal R)
    hs : Eq (s.inf id) I
    hs' : ∀ ⦃J : Ideal R⦄, Membership.mem s J → J.IsPrimary
    t : Finset (Ideal R)
    ht : Eq (t.inf id) I
    ht' : ∀ ⦃J : Ideal R⦄, Membership.mem t J → J.IsPrimary
    ht'' : (↑t).Pairwise (Function.onFun (fun x1 x2 => Ne x1 x2) Ideal.radical)
    ⊢ Exists fun t => And (Eq (t.inf id) I) (And (∀ ⦃J : Ideal R⦄, Membership.mem  …
  -/
  obtain ⟨u, hut, hu, hu'⟩ := decomposition_erase_inf ht
  /-
    case intro.intro.intro.intro.intro.intro
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq (Ideal R)
    I : Ideal R
    s : Finset (Ideal R)
    hs : Eq (s.inf id) I
    hs' : ∀ ⦃J : Ideal R⦄, Membership.mem s J → J.IsPrimary
    t : Finset (Ideal R)
    ht : Eq (t.inf id) I
    ht' : ∀ ⦃J : Ideal R⦄, Membership.mem t J → J.IsPrimary
    ht'' : (↑t).Pairwise (Function.onFun (fun x1 x2 => Ne x1 x2) Ideal.radical)
    u : Finset (Ideal R)
    hut : HasSubset.Subset u t
    hu : Eq (u.inf id) I
    hu' : ∀ ⦃J : Ideal R⦄, Membership.mem u J → Not (LE.le ((u.erase J).inf id) J)
    ⊢ Exists fun t => And (Eq (t.inf id) I) (And (∀ ⦃J : Ideal R⦄, Membership.mem  …
  -/
  exact ⟨u, hu, fun _ hi ↦ ht' (hut hi), ht''.mono hut, hu'⟩
  /-
    🎉 no goals
  -/


lemma IsLasker.minimal [DecidableEq (Ideal R)] (h : IsLasker R) (I : Ideal R) :
    ∃ t : Finset (Ideal R), t.inf id = I ∧ (∀ ⦃J⦄, J ∈ t → J.IsPrimary) ∧
      ((t : Set (Ideal R)).Pairwise ((· ≠ ·) on radical)) ∧
      (∀ ⦃J⦄, J ∈ t → ¬ (t.erase J).inf id ≤ J) := by
  /-
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq (Ideal R)
    h : IsLasker R
    I : Ideal R
    ⊢ Exists fun t => And (Eq (t.inf id) I) (And (∀ ⦃J : Ideal R⦄, Membership.mem  …
  -/
  obtain ⟨s, hs, hs'⟩ := h I
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq (Ideal R)
    h : IsLasker R
    I : Ideal R
    s : Finset (Ideal R)
    hs : Eq (s.inf id) I
    hs' : ∀ ⦃J : Ideal R⦄, Membership.mem s J → J.IsPrimary
    ⊢ Exists fun t => And (Eq (t.inf id) I) (And (∀ ⦃J : Ideal R⦄, Membership.mem  …
  -/
  exact exists_minimal_isPrimary_decomposition_of_isPrimary_decomposition hs hs'
  /-
    🎉 no goals
  -/


lemma _root_.InfIrred.isPrimary {I : Ideal R} (h : InfIrred I) : I.IsPrimary := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I : Ideal R
    h : InfIrred I
    ⊢ I.IsPrimary
  -/
  rw [Ideal.isPrimary_iff]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I : Ideal R
    h : InfIrred I
    ⊢ And (Ne I Top.top) (∀ {x y : R}, Membership.mem I (HMul.hMul x y) → Or (Memb …
  -/
  refine ⟨h.ne_top, fun {a b} hab ↦ ?_⟩
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I : Ideal R
    h : InfIrred I
    a b : R
    hab : Membership.mem I (HMul.hMul a b)
    ⊢ Or (Membership.mem I a) (Membership.mem I.radical b)
  -/
  let f : ℕ → Ideal R := fun n ↦ (I.colon (span {b ^ n}))
  have hf : Monotone f := by
    intro n m hnm
    simp_rw [f]
    exact (Submodule.colon_mono le_rfl (Ideal.span_singleton_le_span_singleton.mpr
      (pow_dvd_pow b hnm)))
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I : Ideal R
    h : InfIrred I
    a b : R
    hab : Membership.mem I (HMul.hMul a b)
    f : Nat → Ideal R := fun n => Submodule.colon I (Ideal.span (Singleton.singlet …
    hf : Monotone f
    ⊢ Or (Membership.mem I a) (Membership.mem I.radical b)
  -/
  obtain ⟨n, hn⟩ := monotone_stabilizes_iff_noetherian.mpr ‹_› ⟨f, hf⟩
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I : Ideal R
    h : InfIrred I
    a b : R
    hab : Membership.mem I (HMul.hMul a b)
    f : Nat → Ideal R := fun n => Submodule.colon I (Ideal.span (Singleton.singlet …
    hf : Monotone f
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq ({ toFun := f, monotone' := hf } n) ({ toFun  …
    ⊢ Or (Membership.mem I a) (Membership.mem I.radical b)
  -/
  rcases h with ⟨-, h⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I : Ideal R
    a b : R
    hab : Membership.mem I (HMul.hMul a b)
    f : Nat → Ideal R := fun n => Submodule.colon I (Ideal.span (Singleton.singlet …
    hf : Monotone f
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq ({ toFun := f, monotone' := hf } n) ({ toFun  …
    h : ∀ ⦃b c : Ideal R⦄, Eq (Min.min b c) I → Or (Eq b I) (Eq c I)
    ⊢ Or (Membership.mem I a) (Membership.mem I.radical b)
  -/
  specialize @h (I.colon (span {b ^ n})) (I + (span {b ^ n})) ?_
    /-
      case intro.intro
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsNoetherianRing R
      I : Ideal R
      a b : R
      hab : Membership.mem I (HMul.hMul a b)
      f : Nat → Ideal R := fun n => Submodule.colon I (Ideal.span (Singleton.singlet …
      hf : Monotone f
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Eq ({ toFun := f, monotone' := hf } n) ({ toFun  …
      h : ∀ ⦃b c : Ideal R⦄, Eq (Min.min b c) I → Or (Eq b I) (Eq c I)
      ⊢ Eq (Min.min (Submodule.colon I (Ideal.span (Singleton.singleton (HPow.hPow b …
    -/
  · refine le_antisymm (fun r ↦ ?_) (le_inf (fun _ ↦ ?_) ?_)
    · simp only [Submodule.add_eq_sup, sup_comm I, mem_inf, mem_colon_singleton,
        mem_span_singleton_sup, and_imp, forall_exists_index]
      /-
        case intro.intro.refine_1
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsNoetherianRing R
        I : Ideal R
        a b : R
        hab : Membership.mem I (HMul.hMul a b)
        f : Nat → Ideal R := fun n => Submodule.colon I (Ideal.span (Singleton.singlet …
        hf : Monotone f
        n : Nat
        hn : ∀ (m : Nat), LE.le n m → Eq ({ toFun := f, monotone' := hf } n) ({ toFun  …
        h : ∀ ⦃b c : Ideal R⦄, Eq (Min.min b c) I → Or (Eq b I) (Eq c I)
        r : R
        ⊢ Membership.mem I (HMul.hMul r (HPow.hPow b n)) → ∀ (x x_1 : R), Membership.m …
      -/
      rintro hrb t s hs rfl
      /-
        case intro.intro.refine_1
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsNoetherianRing R
        I : Ideal R
        a b : R
        hab : Membership.mem I (HMul.hMul a b)
        f : Nat → Ideal R := fun n => Submodule.colon I (Ideal.span (Singleton.singlet …
        hf : Monotone f
        n : Nat
        hn : ∀ (m : Nat), LE.le n m → Eq ({ toFun := f, monotone' := hf } n) ({ toFun  …
        h : ∀ ⦃b c : Ideal R⦄, Eq (Min.min b c) I → Or (Eq b I) (Eq c I)
        t s : R
        hs : Membership.mem I s
        hrb : Membership.mem I (HMul.hMul (HAdd.hAdd (HMul.hMul t (HPow.hPow b n)) s)  …
        ⊢ Membership.mem I (HAdd.hAdd (HMul.hMul t (HPow.hPow b n)) s)
      -/
      refine add_mem ?_ hs
      /-
        case intro.intro.refine_1
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsNoetherianRing R
        I : Ideal R
        a b : R
        hab : Membership.mem I (HMul.hMul a b)
        f : Nat → Ideal R := fun n => Submodule.colon I (Ideal.span (Singleton.singlet …
        hf : Monotone f
        n : Nat
        hn : ∀ (m : Nat), LE.le n m → Eq ({ toFun := f, monotone' := hf } n) ({ toFun  …
        h : ∀ ⦃b c : Ideal R⦄, Eq (Min.min b c) I → Or (Eq b I) (Eq c I)
        t s : R
        hs : Membership.mem I s
        hrb : Membership.mem I (HMul.hMul (HAdd.hAdd (HMul.hMul t (HPow.hPow b n)) s)  …
        ⊢ Membership.mem I (HMul.hMul t (HPow.hPow b n))
      -/
      have := hn (n + n) (by simp)
      /-
        case intro.intro.refine_1
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsNoetherianRing R
        I : Ideal R
        a b : R
        hab : Membership.mem I (HMul.hMul a b)
        f : Nat → Ideal R := fun n => Submodule.colon I (Ideal.span (Singleton.singlet …
        hf : Monotone f
        n : Nat
        hn : ∀ (m : Nat), LE.le n m → Eq ({ toFun := f, monotone' := hf } n) ({ toFun  …
        h : ∀ ⦃b c : Ideal R⦄, Eq (Min.min b c) I → Or (Eq b I) (Eq c I)
        t s : R
        hs : Membership.mem I s
        hrb : Membership.mem I (HMul.hMul (HAdd.hAdd (HMul.hMul t (HPow.hPow b n)) s)  …
        this : Eq ({ toFun := f, monotone' := hf } n) ({ toFun := f, monotone' := hf } …
        ⊢ Membership.mem I (HMul.hMul t (HPow.hPow b n))
      -/
      simp only [OrderHom.coe_mk, f] at this
      /-
        case intro.intro.refine_1
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsNoetherianRing R
        I : Ideal R
        a b : R
        hab : Membership.mem I (HMul.hMul a b)
        f : Nat → Ideal R := fun n => Submodule.colon I (Ideal.span (Singleton.singlet …
        hf : Monotone f
        n : Nat
        hn : ∀ (m : Nat), LE.le n m → Eq ({ toFun := f, monotone' := hf } n) ({ toFun  …
        h : ∀ ⦃b c : Ideal R⦄, Eq (Min.min b c) I → Or (Eq b I) (Eq c I)
        t s : R
        hs : Membership.mem I s
        hrb : Membership.mem I (HMul.hMul (HAdd.hAdd (HMul.hMul t (HPow.hPow b n)) s)  …
        this : Eq (Submodule.colon I (Ideal.span (Singleton.singleton (HPow.hPow b n)) …
        ⊢ Membership.mem I (HMul.hMul t (HPow.hPow b n))
      -/
      rw [add_mul, mul_assoc, ← pow_add] at hrb
      rwa [← mem_colon_singleton, this, mem_colon_singleton,
           ← Ideal.add_mem_iff_left _ (Ideal.mul_mem_right _ _ hs)]
      /-
        case intro.intro.refine_2
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsNoetherianRing R
        I : Ideal R
        a b : R
        hab : Membership.mem I (HMul.hMul a b)
        f : Nat → Ideal R := fun n => Submodule.colon I (Ideal.span (Singleton.singlet …
        hf : Monotone f
        n : Nat
        hn : ∀ (m : Nat), LE.le n m → Eq ({ toFun := f, monotone' := hf } n) ({ toFun  …
        h : ∀ ⦃b c : Ideal R⦄, Eq (Min.min b c) I → Or (Eq b I) (Eq c I)
        x✝ : R
        ⊢ Membership.mem I x✝ → Membership.mem (Submodule.colon I (Ideal.span (Singlet …
      -/
    · simpa only [mem_colon_singleton] using mul_mem_right _ _
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.refine_3
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsNoetherianRing R
        I : Ideal R
        a b : R
        hab : Membership.mem I (HMul.hMul a b)
        f : Nat → Ideal R := fun n => Submodule.colon I (Ideal.span (Singleton.singlet …
        hf : Monotone f
        n : Nat
        hn : ∀ (m : Nat), LE.le n m → Eq ({ toFun := f, monotone' := hf } n) ({ toFun  …
        h : ∀ ⦃b c : Ideal R⦄, Eq (Min.min b c) I → Or (Eq b I) (Eq c I)
        ⊢ LE.le I (HAdd.hAdd I (Ideal.span (Singleton.singleton (HPow.hPow b n))))
      -/
    · simp
      /-
        🎉 no goals
      -/
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    I : Ideal R
    a b : R
    hab : Membership.mem I (HMul.hMul a b)
    f : Nat → Ideal R := fun n => Submodule.colon I (Ideal.span (Singleton.singlet …
    hf : Monotone f
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq ({ toFun := f, monotone' := hf } n) ({ toFun  …
    h : Or (Eq (Submodule.colon I (Ideal.span (Singleton.singleton (HPow.hPow b n) …
    ⊢ Or (Membership.mem I a) (Membership.mem I.radical b)
  -/
  rcases h with (h|h)
  · replace h : I = I.colon (span {b}) := by
      rcases eq_or_ne n 0 with rfl|hn'
      · simpa [f] using hn 1 zero_le_one
      refine le_antisymm ?_ (h.le.trans' (Submodule.colon_mono le_rfl ?_))
      · intro
        simpa only [mem_colon_singleton] using mul_mem_right _ _
      · exact span_singleton_le_span_singleton.mpr (dvd_pow_self b hn')
    /-
      case intro.intro.inl
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsNoetherianRing R
      I : Ideal R
      a b : R
      hab : Membership.mem I (HMul.hMul a b)
      f : Nat → Ideal R := fun n => Submodule.colon I (Ideal.span (Singleton.singlet …
      hf : Monotone f
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Eq ({ toFun := f, monotone' := hf } n) ({ toFun  …
      h : Eq I (Submodule.colon I (Ideal.span (Singleton.singleton b)))
      ⊢ Or (Membership.mem I a) (Membership.mem I.radical b)
    -/
    rw [← mem_colon_singleton, ← h] at hab
    /-
      case intro.intro.inl
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsNoetherianRing R
      I : Ideal R
      a b : R
      hab : Membership.mem I a
      f : Nat → Ideal R := fun n => Submodule.colon I (Ideal.span (Singleton.singlet …
      hf : Monotone f
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Eq ({ toFun := f, monotone' := hf } n) ({ toFun  …
      h : Eq I (Submodule.colon I (Ideal.span (Singleton.singleton b)))
      ⊢ Or (Membership.mem I a) (Membership.mem I.radical b)
    -/
    exact Or.inl hab
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsNoetherianRing R
      I : Ideal R
      a b : R
      hab : Membership.mem I (HMul.hMul a b)
      f : Nat → Ideal R := fun n => Submodule.colon I (Ideal.span (Singleton.singlet …
      hf : Monotone f
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Eq ({ toFun := f, monotone' := hf } n) ({ toFun  …
      h : Eq (HAdd.hAdd I (Ideal.span (Singleton.singleton (HPow.hPow b n)))) I
      ⊢ Or (Membership.mem I a) (Membership.mem I.radical b)
    -/
  · rw [← h]
    /-
      case intro.intro.inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsNoetherianRing R
      I : Ideal R
      a b : R
      hab : Membership.mem I (HMul.hMul a b)
      f : Nat → Ideal R := fun n => Submodule.colon I (Ideal.span (Singleton.singlet …
      hf : Monotone f
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Eq ({ toFun := f, monotone' := hf } n) ({ toFun  …
      h : Eq (HAdd.hAdd I (Ideal.span (Singleton.singleton (HPow.hPow b n)))) I
      ⊢ Or (Membership.mem (HAdd.hAdd I (Ideal.span (Singleton.singleton (HPow.hPow  …
    -/
    refine Or.inr ⟨n, ?_⟩
    /-
      case intro.intro.inr
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsNoetherianRing R
      I : Ideal R
      a b : R
      hab : Membership.mem I (HMul.hMul a b)
      f : Nat → Ideal R := fun n => Submodule.colon I (Ideal.span (Singleton.singlet …
      hf : Monotone f
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Eq ({ toFun := f, monotone' := hf } n) ({ toFun  …
      h : Eq (HAdd.hAdd I (Ideal.span (Singleton.singleton (HPow.hPow b n)))) I
      ⊢ Membership.mem (HAdd.hAdd I (Ideal.span (Singleton.singleton (HPow.hPow b n) …
    -/
    simpa using mem_sup_right (mem_span_singleton_self _)
    /-
      🎉 no goals
    -/


variable (R) in
/-- The Lasker--Noether theorem: every ideal in a Noetherian ring admits a decomposition into
  primary ideals. -/
lemma isLasker : IsLasker R := fun I ↦
  (exists_infIrred_decomposition I).imp fun _ h ↦ h.imp_right fun h' _ ht ↦ (h' ht).isPrimary



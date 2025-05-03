/-- A proper ideal `I` is primary as a submodule. -/
abbrev IsPrimary (I : Ideal R) : Prop :=
  Submodule.IsPrimary I


/-- A proper ideal `I` is primary iff `xy ∈ I` implies `x ∈ I` or `y ∈ radical I`. -/
lemma isPrimary_iff {I : Ideal R} :
    I.IsPrimary ↔ I ≠ ⊤ ∧ ∀ {x y : R}, x * y ∈ I → x ∈ I ∨ y ∈ radical I := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    I : Ideal R
    ⊢ Iff I.IsPrimary (And (Ne I Top.top) (∀ {x y : R}, Membership.mem I (HMul.hMu …
  -/
  rw [IsPrimary, Submodule.IsPrimary, forall_comm]
  simp only [mul_comm, mem_radical_iff, and_congr_right_iff,
    ← Submodule.ideal_span_singleton_smul, smul_eq_mul, mul_top, span_singleton_le_iff_mem]


theorem IsPrime.isPrimary {I : Ideal R} (hi : IsPrime I) : I.IsPrimary :=
  isPrimary_iff.mpr
  ⟨hi.1, fun {_ _} hxy => (hi.mem_or_mem hxy).imp id fun hyi => le_radical hyi⟩


theorem isPrime_radical {I : Ideal R} (hi : I.IsPrimary) : IsPrime (radical I) :=
  ⟨mt radical_eq_top.1 hi.1,
   fun {x y} ⟨m, hxy⟩ => by
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      I : Ideal R
      hi : I.IsPrimary
      x y : R
      x✝ : Membership.mem I.radical (HMul.hMul x y)
      m : Nat
      hxy : Membership.mem I (HPow.hPow (HMul.hMul x y) m)
      ⊢ Or (Membership.mem I.radical x) (Membership.mem I.radical y)
    -/
    rw [mul_pow] at hxy; cases' (isPrimary_iff.mp hi).2 hxy with h h
      /-
        case inl
        R : Type u_1
        inst✝ : CommSemiring R
        I : Ideal R
        hi : I.IsPrimary
        x y : R
        x✝ : Membership.mem I.radical (HMul.hMul x y)
        m : Nat
        hxy : Membership.mem I (HMul.hMul (HPow.hPow x m) (HPow.hPow y m))
        h : Membership.mem I (HPow.hPow x m)
        ⊢ Or (Membership.mem I.radical x) (Membership.mem I.radical y)
      -/
    · exact Or.inl ⟨m, h⟩
      /-
        🎉 no goals
      -/
      /-
        case inr
        R : Type u_1
        inst✝ : CommSemiring R
        I : Ideal R
        hi : I.IsPrimary
        x y : R
        x✝ : Membership.mem I.radical (HMul.hMul x y)
        m : Nat
        hxy : Membership.mem I (HMul.hMul (HPow.hPow x m) (HPow.hPow y m))
        h : Membership.mem I.radical (HPow.hPow y m)
        ⊢ Or (Membership.mem I.radical x) (Membership.mem I.radical y)
      -/
    · exact Or.inr (mem_radical_of_pow_mem h)⟩
      /-
        🎉 no goals
      -/


theorem isPrimary_inf {I J : Ideal R} (hi : I.IsPrimary) (hj : J.IsPrimary)
    (hij : radical I = radical J) : (I ⊓ J).IsPrimary :=
  isPrimary_iff.mpr
  ⟨ne_of_lt <| lt_of_le_of_lt inf_le_left (lt_top_iff_ne_top.2 hi.1),
   fun {x y} ⟨hxyi, hxyj⟩ => by
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      I J : Ideal R
      hi : I.IsPrimary
      hj : J.IsPrimary
      hij : Eq I.radical J.radical
      x y : R
      x✝ : Membership.mem (Min.min I J) (HMul.hMul x y)
      hxyi : Membership.mem (↑I) (HMul.hMul x y)
      hxyj : Membership.mem (↑J) (HMul.hMul x y)
      ⊢ Or (Membership.mem (Min.min I J) x) (Membership.mem (Min.min I J).radical y)
    -/
    rw [radical_inf, hij, inf_idem]
    /-
      R : Type u_1
      inst✝ : CommSemiring R
      I J : Ideal R
      hi : I.IsPrimary
      hj : J.IsPrimary
      hij : Eq I.radical J.radical
      x y : R
      x✝ : Membership.mem (Min.min I J) (HMul.hMul x y)
      hxyi : Membership.mem (↑I) (HMul.hMul x y)
      hxyj : Membership.mem (↑J) (HMul.hMul x y)
      ⊢ Or (Membership.mem (Min.min I J) x) (Membership.mem J.radical y)
    -/
    cases' (isPrimary_iff.mp hi).2 hxyi with hxi hyi
      /-
        case inl
        R : Type u_1
        inst✝ : CommSemiring R
        I J : Ideal R
        hi : I.IsPrimary
        hj : J.IsPrimary
        hij : Eq I.radical J.radical
        x y : R
        x✝ : Membership.mem (Min.min I J) (HMul.hMul x y)
        hxyi : Membership.mem (↑I) (HMul.hMul x y)
        hxyj : Membership.mem (↑J) (HMul.hMul x y)
        hxi : Membership.mem I x
        ⊢ Or (Membership.mem (Min.min I J) x) (Membership.mem J.radical y)
      -/
    · cases' (isPrimary_iff.mp hj).2 hxyj with hxj hyj
        /-
          case inl.inl
          R : Type u_1
          inst✝ : CommSemiring R
          I J : Ideal R
          hi : I.IsPrimary
          hj : J.IsPrimary
          hij : Eq I.radical J.radical
          x y : R
          x✝ : Membership.mem (Min.min I J) (HMul.hMul x y)
          hxyi : Membership.mem (↑I) (HMul.hMul x y)
          hxyj : Membership.mem (↑J) (HMul.hMul x y)
          hxi : Membership.mem I x
          hxj : Membership.mem J x
          ⊢ Or (Membership.mem (Min.min I J) x) (Membership.mem J.radical y)
        -/
      · exact Or.inl ⟨hxi, hxj⟩
        /-
          🎉 no goals
        -/
        /-
          case inl.inr
          R : Type u_1
          inst✝ : CommSemiring R
          I J : Ideal R
          hi : I.IsPrimary
          hj : J.IsPrimary
          hij : Eq I.radical J.radical
          x y : R
          x✝ : Membership.mem (Min.min I J) (HMul.hMul x y)
          hxyi : Membership.mem (↑I) (HMul.hMul x y)
          hxyj : Membership.mem (↑J) (HMul.hMul x y)
          hxi : Membership.mem I x
          hyj : Membership.mem J.radical y
          ⊢ Or (Membership.mem (Min.min I J) x) (Membership.mem J.radical y)
        -/
      · exact Or.inr hyj
        /-
          🎉 no goals
        -/
      /-
        case inr
        R : Type u_1
        inst✝ : CommSemiring R
        I J : Ideal R
        hi : I.IsPrimary
        hj : J.IsPrimary
        hij : Eq I.radical J.radical
        x y : R
        x✝ : Membership.mem (Min.min I J) (HMul.hMul x y)
        hxyi : Membership.mem (↑I) (HMul.hMul x y)
        hxyj : Membership.mem (↑J) (HMul.hMul x y)
        hyi : Membership.mem I.radical y
        ⊢ Or (Membership.mem (Min.min I J) x) (Membership.mem J.radical y)
      -/
    · rw [hij] at hyi
      /-
        case inr
        R : Type u_1
        inst✝ : CommSemiring R
        I J : Ideal R
        hi : I.IsPrimary
        hj : J.IsPrimary
        hij : Eq I.radical J.radical
        x y : R
        x✝ : Membership.mem (Min.min I J) (HMul.hMul x y)
        hxyi : Membership.mem (↑I) (HMul.hMul x y)
        hxyj : Membership.mem (↑J) (HMul.hMul x y)
        hyi : Membership.mem J.radical y
        ⊢ Or (Membership.mem (Min.min I J) x) (Membership.mem J.radical y)
      -/
      exact Or.inr hyi⟩
      /-
        🎉 no goals
      -/


open Finset in
lemma isPrimary_finset_inf {ι} {s : Finset ι} {f : ι → Ideal R} {i : ι} (hi : i ∈ s)
    (hs : ∀ ⦃y⦄, y ∈ s → (f y).IsPrimary)
    (hs' : ∀ ⦃y⦄, y ∈ s → (f y).radical = (f i).radical) :
    IsPrimary (s.inf f) := by
  classical
  induction s using Finset.induction_on generalizing i with
  | empty => simp at hi
  | @insert a s ha IH =>
    rcases s.eq_empty_or_nonempty with rfl|⟨y, hy⟩
    · simp only [insert_emptyc_eq, mem_singleton] at hi
      simpa [hi] using hs
    simp only [inf_insert]
    have H : ∀ ⦃x : ι⦄, x ∈ s → (f x).radical = (f y).radical := by
      intro x hx
      rw [hs' (mem_insert_of_mem hx), hs' (mem_insert_of_mem hy)]
    refine isPrimary_inf (hs (by simp)) (IH hy ?_ H) ?_
    · intro x hx
      exact hs (by simp [hx])
    · rw [radical_finset_inf hy H, hs' (mem_insert_self _ _), hs' (mem_insert_of_mem hy)]



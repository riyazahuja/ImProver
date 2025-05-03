/-- **Nakayama's Lemma**. Atiyah-Macdonald 2.5, Eisenbud 4.7, Matsumura 2.2,
[Stacks 00DV](https://stacks.math.columbia.edu/tag/00DV) -/
theorem exists_sub_one_mem_and_smul_eq_zero_of_fg_of_le_smul {R : Type*} [CommRing R] {M : Type*}
    [AddCommGroup M] [Module R M] (I : Ideal R) (N : Submodule R M) (hn : N.FG) (hin : N ≤ I • N) :
    ∃ r : R, r - 1 ∈ I ∧ ∀ n ∈ N, r • n = (0 : M) := by
  /-
    R : Type u_3
    inst✝² : CommRing R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    hn : N.FG
    hin : LE.le N (HSMul.hSMul I N)
    ⊢ Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (∀ (n : M), Membershi …
  -/
  rw [fg_def] at hn
  /-
    R : Type u_3
    inst✝² : CommRing R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    hn : Exists fun S => And S.Finite (Eq (Submodule.span R S) N)
    hin : LE.le N (HSMul.hSMul I N)
    ⊢ Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (∀ (n : M), Membershi …
  -/
  rcases hn with ⟨s, hfs, hs⟩
  have : ∃ r : R, r - 1 ∈ I ∧ N ≤ (I • span R s).comap (LinearMap.lsmul R M r) ∧ s ⊆ N := by
    refine ⟨1, ?_, ?_, ?_⟩
    · rw [sub_self]
      exact I.zero_mem
    · rw [hs]
      intro n hn
      rw [mem_comap]
      change (1 : R) • n ∈ I • N
      rw [one_smul]
      exact hin hn
    · rw [← span_le, hs]
  /-
    case intro.intro
    R : Type u_3
    inst✝² : CommRing R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    hin : LE.le N (HSMul.hSMul I N)
    s : Set M
    hfs : s.Finite
    hs : Eq (Submodule.span R s) N
    this : Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (S …
    ⊢ Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (∀ (n : M), Membershi …
  -/
  clear hin hs
  /-
    case intro.intro
    R : Type u_3
    inst✝² : CommRing R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    s : Set M
    hfs : s.Finite
    this : Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (S …
    ⊢ Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (∀ (n : M), Membershi …
  -/
  revert this
  /-
    case intro.intro
    R : Type u_3
    inst✝² : CommRing R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    s : Set M
    hfs : s.Finite
    ⊢ (Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Submo …
  -/
  refine Set.Finite.dinduction_on _ hfs (fun H => ?_) @fun i s _ _ ih H => ?_
    /-
      case intro.intro.refine_1
      R : Type u_3
      inst✝² : CommRing R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N : Submodule R M
      s : Set M
      hfs : s.Finite
      H : Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Subm …
      ⊢ Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (∀ (n : M), Membershi …
    -/
  · rcases H with ⟨r, hr1, hrn, _⟩
    /-
      case intro.intro.refine_1.intro.intro.intro
      R : Type u_3
      inst✝² : CommRing R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N : Submodule R M
      s : Set M
      hfs : s.Finite
      r : R
      hr1 : Membership.mem I (HSub.hSub r 1)
      hrn : LE.le N (Submodule.comap ((LinearMap.lsmul R M) r) (HSMul.hSMul I (Submo …
      right✝ : HasSubset.Subset EmptyCollection.emptyCollection ↑N
      ⊢ Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (∀ (n : M), Membershi …
    -/
    refine ⟨r, hr1, fun n hn => ?_⟩
    /-
      case intro.intro.refine_1.intro.intro.intro
      R : Type u_3
      inst✝² : CommRing R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N : Submodule R M
      s : Set M
      hfs : s.Finite
      r : R
      hr1 : Membership.mem I (HSub.hSub r 1)
      hrn : LE.le N (Submodule.comap ((LinearMap.lsmul R M) r) (HSMul.hSMul I (Submo …
      right✝ : HasSubset.Subset EmptyCollection.emptyCollection ↑N
      n : M
      hn : Membership.mem N n
      ⊢ Eq (HSMul.hSMul r n) 0
    -/
    specialize hrn hn
    /-
      case intro.intro.refine_1.intro.intro.intro
      R : Type u_3
      inst✝² : CommRing R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N : Submodule R M
      s : Set M
      hfs : s.Finite
      r : R
      hr1 : Membership.mem I (HSub.hSub r 1)
      right✝ : HasSubset.Subset EmptyCollection.emptyCollection ↑N
      n : M
      hn : Membership.mem N n
      hrn : Membership.mem (Submodule.comap ((LinearMap.lsmul R M) r) (HSMul.hSMul I …
      ⊢ Eq (HSMul.hSMul r n) 0
    -/
    rwa [mem_comap, span_empty, smul_bot, mem_bot] at hrn
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.refine_2
    R : Type u_3
    inst✝² : CommRing R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    s✝ : Set M
    hfs : s✝.Finite
    i : M
    s : Set M
    x✝¹ : Not (Membership.mem s i)
    x✝ : s.Finite
    ih : (Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Su …
    H : Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Subm …
    ⊢ Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (∀ (n : M), Membershi …
  -/
  apply ih
  /-
    case intro.intro.refine_2
    R : Type u_3
    inst✝² : CommRing R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    s✝ : Set M
    hfs : s✝.Finite
    i : M
    s : Set M
    x✝¹ : Not (Membership.mem s i)
    x✝ : s.Finite
    ih : (Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Su …
    H : Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Subm …
    ⊢ Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Submod …
  -/
  rcases H with ⟨r, hr1, hrn, hs⟩
  /-
    case intro.intro.refine_2.intro.intro.intro
    R : Type u_3
    inst✝² : CommRing R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    s✝ : Set M
    hfs : s✝.Finite
    i : M
    s : Set M
    x✝¹ : Not (Membership.mem s i)
    x✝ : s.Finite
    ih : (Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Su …
    r : R
    hr1 : Membership.mem I (HSub.hSub r 1)
    hrn : LE.le N (Submodule.comap ((LinearMap.lsmul R M) r) (HSMul.hSMul I (Submo …
    hs : HasSubset.Subset (Insert.insert i s) ↑N
    ⊢ Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Submod …
  -/
  rw [← Set.singleton_union, span_union, smul_sup] at hrn
  /-
    case intro.intro.refine_2.intro.intro.intro
    R : Type u_3
    inst✝² : CommRing R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    s✝ : Set M
    hfs : s✝.Finite
    i : M
    s : Set M
    x✝¹ : Not (Membership.mem s i)
    x✝ : s.Finite
    ih : (Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Su …
    r : R
    hr1 : Membership.mem I (HSub.hSub r 1)
    hrn : LE.le N (Submodule.comap ((LinearMap.lsmul R M) r) (Max.max (HSMul.hSMul …
    hs : HasSubset.Subset (Insert.insert i s) ↑N
    ⊢ Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Submod …
  -/
  rw [Set.insert_subset_iff] at hs
  have : ∃ c : R, c - 1 ∈ I ∧ c • i ∈ I • span R s := by
    specialize hrn hs.1
    rw [mem_comap, mem_sup] at hrn
    rcases hrn with ⟨y, hy, z, hz, hyz⟩
    dsimp at hyz
    rw [mem_smul_span_singleton] at hy
    rcases hy with ⟨c, hci, rfl⟩
    use r - c
    constructor
    · rw [sub_right_comm]
      exact I.sub_mem hr1 hci
    · rw [sub_smul, ← hyz, add_sub_cancel_left]
      exact hz
  /-
    case intro.intro.refine_2.intro.intro.intro
    R : Type u_3
    inst✝² : CommRing R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    s✝ : Set M
    hfs : s✝.Finite
    i : M
    s : Set M
    x✝¹ : Not (Membership.mem s i)
    x✝ : s.Finite
    ih : (Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Su …
    r : R
    hr1 : Membership.mem I (HSub.hSub r 1)
    hrn : LE.le N (Submodule.comap ((LinearMap.lsmul R M) r) (Max.max (HSMul.hSMul …
    hs : And (Membership.mem (↑N) i) (HasSubset.Subset s ↑N)
    this : Exists fun c => And (Membership.mem I (HSub.hSub c 1)) (Membership.mem  …
    ⊢ Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Submod …
  -/
  rcases this with ⟨c, hc1, hci⟩
  /-
    case intro.intro.refine_2.intro.intro.intro.intro.intro
    R : Type u_3
    inst✝² : CommRing R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    s✝ : Set M
    hfs : s✝.Finite
    i : M
    s : Set M
    x✝¹ : Not (Membership.mem s i)
    x✝ : s.Finite
    ih : (Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Su …
    r : R
    hr1 : Membership.mem I (HSub.hSub r 1)
    hrn : LE.le N (Submodule.comap ((LinearMap.lsmul R M) r) (Max.max (HSMul.hSMul …
    hs : And (Membership.mem (↑N) i) (HasSubset.Subset s ↑N)
    c : R
    hc1 : Membership.mem I (HSub.hSub c 1)
    hci : Membership.mem (HSMul.hSMul I (Submodule.span R s)) (HSMul.hSMul c i)
    ⊢ Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Submod …
  -/
  refine ⟨c * r, ?_, ?_, hs.2⟩
    /-
      case intro.intro.refine_2.intro.intro.intro.intro.intro.refine_1
      R : Type u_3
      inst✝² : CommRing R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N : Submodule R M
      s✝ : Set M
      hfs : s✝.Finite
      i : M
      s : Set M
      x✝¹ : Not (Membership.mem s i)
      x✝ : s.Finite
      ih : (Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Su …
      r : R
      hr1 : Membership.mem I (HSub.hSub r 1)
      hrn : LE.le N (Submodule.comap ((LinearMap.lsmul R M) r) (Max.max (HSMul.hSMul …
      hs : And (Membership.mem (↑N) i) (HasSubset.Subset s ↑N)
      c : R
      hc1 : Membership.mem I (HSub.hSub c 1)
      hci : Membership.mem (HSMul.hSMul I (Submodule.span R s)) (HSMul.hSMul c i)
      ⊢ Membership.mem I (HSub.hSub (HMul.hMul c r) 1)
    -/
  · simpa only [mul_sub, mul_one, sub_add_sub_cancel] using I.add_mem (I.mul_mem_left c hr1) hc1
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2.intro.intro.intro.intro.intro.refine_2
      R : Type u_3
      inst✝² : CommRing R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N : Submodule R M
      s✝ : Set M
      hfs : s✝.Finite
      i : M
      s : Set M
      x✝¹ : Not (Membership.mem s i)
      x✝ : s.Finite
      ih : (Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Su …
      r : R
      hr1 : Membership.mem I (HSub.hSub r 1)
      hrn : LE.le N (Submodule.comap ((LinearMap.lsmul R M) r) (Max.max (HSMul.hSMul …
      hs : And (Membership.mem (↑N) i) (HasSubset.Subset s ↑N)
      c : R
      hc1 : Membership.mem I (HSub.hSub c 1)
      hci : Membership.mem (HSMul.hSMul I (Submodule.span R s)) (HSMul.hSMul c i)
      ⊢ LE.le N (Submodule.comap ((LinearMap.lsmul R M) (HMul.hMul c r)) (HSMul.hSMu …
    -/
  · intro n hn
    /-
      case intro.intro.refine_2.intro.intro.intro.intro.intro.refine_2
      R : Type u_3
      inst✝² : CommRing R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N : Submodule R M
      s✝ : Set M
      hfs : s✝.Finite
      i : M
      s : Set M
      x✝¹ : Not (Membership.mem s i)
      x✝ : s.Finite
      ih : (Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Su …
      r : R
      hr1 : Membership.mem I (HSub.hSub r 1)
      hrn : LE.le N (Submodule.comap ((LinearMap.lsmul R M) r) (Max.max (HSMul.hSMul …
      hs : And (Membership.mem (↑N) i) (HasSubset.Subset s ↑N)
      c : R
      hc1 : Membership.mem I (HSub.hSub c 1)
      hci : Membership.mem (HSMul.hSMul I (Submodule.span R s)) (HSMul.hSMul c i)
      n : M
      hn : Membership.mem N n
      ⊢ Membership.mem (Submodule.comap ((LinearMap.lsmul R M) (HMul.hMul c r)) (HSM …
    -/
    specialize hrn hn
    /-
      case intro.intro.refine_2.intro.intro.intro.intro.intro.refine_2
      R : Type u_3
      inst✝² : CommRing R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N : Submodule R M
      s✝ : Set M
      hfs : s✝.Finite
      i : M
      s : Set M
      x✝¹ : Not (Membership.mem s i)
      x✝ : s.Finite
      ih : (Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Su …
      r : R
      hr1 : Membership.mem I (HSub.hSub r 1)
      hs : And (Membership.mem (↑N) i) (HasSubset.Subset s ↑N)
      c : R
      hc1 : Membership.mem I (HSub.hSub c 1)
      hci : Membership.mem (HSMul.hSMul I (Submodule.span R s)) (HSMul.hSMul c i)
      n : M
      hn : Membership.mem N n
      hrn : Membership.mem (Submodule.comap ((LinearMap.lsmul R M) r) (Max.max (HSMu …
      ⊢ Membership.mem (Submodule.comap ((LinearMap.lsmul R M) (HMul.hMul c r)) (HSM …
    -/
    rw [mem_comap, mem_sup] at hrn
    /-
      case intro.intro.refine_2.intro.intro.intro.intro.intro.refine_2
      R : Type u_3
      inst✝² : CommRing R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N : Submodule R M
      s✝ : Set M
      hfs : s✝.Finite
      i : M
      s : Set M
      x✝¹ : Not (Membership.mem s i)
      x✝ : s.Finite
      ih : (Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Su …
      r : R
      hr1 : Membership.mem I (HSub.hSub r 1)
      hs : And (Membership.mem (↑N) i) (HasSubset.Subset s ↑N)
      c : R
      hc1 : Membership.mem I (HSub.hSub c 1)
      hci : Membership.mem (HSMul.hSMul I (Submodule.span R s)) (HSMul.hSMul c i)
      n : M
      hn : Membership.mem N n
      hrn : Exists fun y => And (Membership.mem (HSMul.hSMul I (Submodule.span R (Si …
      ⊢ Membership.mem (Submodule.comap ((LinearMap.lsmul R M) (HMul.hMul c r)) (HSM …
    -/
    rcases hrn with ⟨y, hy, z, hz, hyz⟩
    /-
      case intro.intro.refine_2.intro.intro.intro.intro.intro.refine_2.intro.intro.i …
      R : Type u_3
      inst✝² : CommRing R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N : Submodule R M
      s✝ : Set M
      hfs : s✝.Finite
      i : M
      s : Set M
      x✝¹ : Not (Membership.mem s i)
      x✝ : s.Finite
      ih : (Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Su …
      r : R
      hr1 : Membership.mem I (HSub.hSub r 1)
      hs : And (Membership.mem (↑N) i) (HasSubset.Subset s ↑N)
      c : R
      hc1 : Membership.mem I (HSub.hSub c 1)
      hci : Membership.mem (HSMul.hSMul I (Submodule.span R s)) (HSMul.hSMul c i)
      n : M
      hn : Membership.mem N n
      y : M
      hy : Membership.mem (HSMul.hSMul I (Submodule.span R (Singleton.singleton i))) y
      z : M
      hz : Membership.mem (HSMul.hSMul I (Submodule.span R s)) z
      hyz : Eq (HAdd.hAdd y z) (((LinearMap.lsmul R M) r) n)
      ⊢ Membership.mem (Submodule.comap ((LinearMap.lsmul R M) (HMul.hMul c r)) (HSM …
    -/
    dsimp at hyz
    /-
      case intro.intro.refine_2.intro.intro.intro.intro.intro.refine_2.intro.intro.i …
      R : Type u_3
      inst✝² : CommRing R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N : Submodule R M
      s✝ : Set M
      hfs : s✝.Finite
      i : M
      s : Set M
      x✝¹ : Not (Membership.mem s i)
      x✝ : s.Finite
      ih : (Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Su …
      r : R
      hr1 : Membership.mem I (HSub.hSub r 1)
      hs : And (Membership.mem (↑N) i) (HasSubset.Subset s ↑N)
      c : R
      hc1 : Membership.mem I (HSub.hSub c 1)
      hci : Membership.mem (HSMul.hSMul I (Submodule.span R s)) (HSMul.hSMul c i)
      n : M
      hn : Membership.mem N n
      y : M
      hy : Membership.mem (HSMul.hSMul I (Submodule.span R (Singleton.singleton i))) y
      z : M
      hz : Membership.mem (HSMul.hSMul I (Submodule.span R s)) z
      hyz : Eq (HAdd.hAdd y z) (HSMul.hSMul r n)
      ⊢ Membership.mem (Submodule.comap ((LinearMap.lsmul R M) (HMul.hMul c r)) (HSM …
    -/
    rw [mem_smul_span_singleton] at hy
    /-
      case intro.intro.refine_2.intro.intro.intro.intro.intro.refine_2.intro.intro.i …
      R : Type u_3
      inst✝² : CommRing R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N : Submodule R M
      s✝ : Set M
      hfs : s✝.Finite
      i : M
      s : Set M
      x✝¹ : Not (Membership.mem s i)
      x✝ : s.Finite
      ih : (Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Su …
      r : R
      hr1 : Membership.mem I (HSub.hSub r 1)
      hs : And (Membership.mem (↑N) i) (HasSubset.Subset s ↑N)
      c : R
      hc1 : Membership.mem I (HSub.hSub c 1)
      hci : Membership.mem (HSMul.hSMul I (Submodule.span R s)) (HSMul.hSMul c i)
      n : M
      hn : Membership.mem N n
      y : M
      hy : Exists fun y_1 => And (Membership.mem I y_1) (Eq (HSMul.hSMul y_1 i) y)
      z : M
      hz : Membership.mem (HSMul.hSMul I (Submodule.span R s)) z
      hyz : Eq (HAdd.hAdd y z) (HSMul.hSMul r n)
      ⊢ Membership.mem (Submodule.comap ((LinearMap.lsmul R M) (HMul.hMul c r)) (HSM …
    -/
    rcases hy with ⟨d, _, rfl⟩
    /-
      case intro.intro.refine_2.intro.intro.intro.intro.intro.refine_2.intro.intro.i …
      R : Type u_3
      inst✝² : CommRing R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N : Submodule R M
      s✝ : Set M
      hfs : s✝.Finite
      i : M
      s : Set M
      x✝¹ : Not (Membership.mem s i)
      x✝ : s.Finite
      ih : (Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Su …
      r : R
      hr1 : Membership.mem I (HSub.hSub r 1)
      hs : And (Membership.mem (↑N) i) (HasSubset.Subset s ↑N)
      c : R
      hc1 : Membership.mem I (HSub.hSub c 1)
      hci : Membership.mem (HSMul.hSMul I (Submodule.span R s)) (HSMul.hSMul c i)
      n : M
      hn : Membership.mem N n
      z : M
      hz : Membership.mem (HSMul.hSMul I (Submodule.span R s)) z
      d : R
      left✝ : Membership.mem I d
      hyz : Eq (HAdd.hAdd (HSMul.hSMul d i) z) (HSMul.hSMul r n)
      ⊢ Membership.mem (Submodule.comap ((LinearMap.lsmul R M) (HMul.hMul c r)) (HSM …
    -/
    simp only [mem_comap, LinearMap.lsmul_apply]
    /-
      case intro.intro.refine_2.intro.intro.intro.intro.intro.refine_2.intro.intro.i …
      R : Type u_3
      inst✝² : CommRing R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N : Submodule R M
      s✝ : Set M
      hfs : s✝.Finite
      i : M
      s : Set M
      x✝¹ : Not (Membership.mem s i)
      x✝ : s.Finite
      ih : (Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Su …
      r : R
      hr1 : Membership.mem I (HSub.hSub r 1)
      hs : And (Membership.mem (↑N) i) (HasSubset.Subset s ↑N)
      c : R
      hc1 : Membership.mem I (HSub.hSub c 1)
      hci : Membership.mem (HSMul.hSMul I (Submodule.span R s)) (HSMul.hSMul c i)
      n : M
      hn : Membership.mem N n
      z : M
      hz : Membership.mem (HSMul.hSMul I (Submodule.span R s)) z
      d : R
      left✝ : Membership.mem I d
      hyz : Eq (HAdd.hAdd (HSMul.hSMul d i) z) (HSMul.hSMul r n)
      ⊢ Membership.mem (HSMul.hSMul I (Submodule.span R s)) (HSMul.hSMul (HMul.hMul  …
    -/
    rw [mul_smul, ← hyz, smul_add, smul_smul, mul_comm, mul_smul]
    /-
      case intro.intro.refine_2.intro.intro.intro.intro.intro.refine_2.intro.intro.i …
      R : Type u_3
      inst✝² : CommRing R
      M : Type u_4
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      N : Submodule R M
      s✝ : Set M
      hfs : s✝.Finite
      i : M
      s : Set M
      x✝¹ : Not (Membership.mem s i)
      x✝ : s.Finite
      ih : (Exists fun r => And (Membership.mem I (HSub.hSub r 1)) (And (LE.le N (Su …
      r : R
      hr1 : Membership.mem I (HSub.hSub r 1)
      hs : And (Membership.mem (↑N) i) (HasSubset.Subset s ↑N)
      c : R
      hc1 : Membership.mem I (HSub.hSub c 1)
      hci : Membership.mem (HSMul.hSMul I (Submodule.span R s)) (HSMul.hSMul c i)
      n : M
      hn : Membership.mem N n
      z : M
      hz : Membership.mem (HSMul.hSMul I (Submodule.span R s)) z
      d : R
      left✝ : Membership.mem I d
      hyz : Eq (HAdd.hAdd (HSMul.hSMul d i) z) (HSMul.hSMul r n)
      ⊢ Membership.mem (HSMul.hSMul I (Submodule.span R s)) (HAdd.hAdd (HSMul.hSMul  …
    -/
    exact add_mem (smul_mem _ _ hci) (smul_mem _ _ hz)
    /-
      🎉 no goals
    -/


theorem exists_mem_and_smul_eq_self_of_fg_of_le_smul {R : Type*} [CommRing R] {M : Type*}
    [AddCommGroup M] [Module R M] (I : Ideal R) (N : Submodule R M) (hn : N.FG) (hin : N ≤ I • N) :
    ∃ r ∈ I, ∀ n ∈ N, r • n = n := by
  /-
    R : Type u_3
    inst✝² : CommRing R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    hn : N.FG
    hin : LE.le N (HSMul.hSMul I N)
    ⊢ Exists fun r => And (Membership.mem I r) (∀ (n : M), Membership.mem N n → Eq …
  -/
  obtain ⟨r, hr, hr'⟩ := exists_sub_one_mem_and_smul_eq_zero_of_fg_of_le_smul I N hn hin
  /-
    case intro.intro
    R : Type u_3
    inst✝² : CommRing R
    M : Type u_4
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    hn : N.FG
    hin : LE.le N (HSMul.hSMul I N)
    r : R
    hr : Membership.mem I (HSub.hSub r 1)
    hr' : ∀ (n : M), Membership.mem N n → Eq (HSMul.hSMul r n) 0
    ⊢ Exists fun r => And (Membership.mem I r) (∀ (n : M), Membership.mem N n → Eq …
  -/
  exact ⟨-(r - 1), I.neg_mem hr, fun n hn => by simpa [sub_smul] using hr' n hn⟩
  /-
    🎉 no goals
  -/



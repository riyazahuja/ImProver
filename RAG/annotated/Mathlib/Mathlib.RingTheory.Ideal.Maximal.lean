/-- An ideal is maximal if it is maximal in the collection of proper ideals. -/
class IsMaximal (I : Ideal α) : Prop where
  /-- The maximal ideal is a coatom in the ordering on ideals; that is, it is not the entire ring,
  and there are no other proper ideals strictly containing it. -/
  out : IsCoatom I


theorem isMaximal_def {I : Ideal α} : I.IsMaximal ↔ IsCoatom I :=
  ⟨fun h => h.1, fun h => ⟨h⟩⟩


theorem IsMaximal.ne_top {I : Ideal α} (h : I.IsMaximal) : I ≠ ⊤ :=
  (isMaximal_def.1 h).1


theorem isMaximal_iff {I : Ideal α} :
    I.IsMaximal ↔ (1 : α) ∉ I ∧ ∀ (J : Ideal α) (x), I ≤ J → x ∉ I → x ∈ J → (1 : α) ∈ J :=
  isMaximal_def.trans <|
    and_congr I.ne_top_iff_one <|
      forall_congr' fun J => by
        /-
          α : Type u
          inst✝ : Semiring α
          I J : Ideal α
          ⊢ Iff (LT.lt I J → Eq J Top.top) (∀ (x : α), LE.le I J → Not (Membership.mem I …
        -/
        rw [lt_iff_le_not_le]
        exact
          ⟨fun H x h hx₁ hx₂ => J.eq_top_iff_one.1 <| H ⟨h, not_subset.2 ⟨_, hx₂, hx₁⟩⟩,
            fun H ⟨h₁, h₂⟩ =>
            let ⟨x, xJ, xI⟩ := not_subset.1 h₂
            J.eq_top_iff_one.2 <| H x h₁ xI xJ⟩


theorem IsMaximal.eq_of_le {I J : Ideal α} (hI : I.IsMaximal) (hJ : J ≠ ⊤) (IJ : I ≤ J) : I = J :=
  eq_iff_le_not_lt.2 ⟨IJ, fun h => hJ (hI.1.2 _ h)⟩


instance : IsCoatomic (Ideal α) := by
  /-
    α : Type u
    β : Type v
    F : Type w
    inst✝ : Semiring α
    I : Ideal α
    a b : α
    ⊢ IsCoatomic (Ideal α)
  -/
  apply CompleteLattice.coatomic_of_top_compact
  /-
    case h
    α : Type u
    β : Type v
    F : Type w
    inst✝ : Semiring α
    I : Ideal α
    a b : α
    ⊢ CompleteLattice.IsCompactElement Top.top
  -/
  rw [← span_singleton_one]
  /-
    case h
    α : Type u
    β : Type v
    F : Type w
    inst✝ : Semiring α
    I : Ideal α
    a b : α
    ⊢ CompleteLattice.IsCompactElement (Ideal.span (Singleton.singleton 1))
  -/
  exact Submodule.singleton_span_isCompactElement 1
  /-
    🎉 no goals
  -/


theorem IsMaximal.coprime_of_ne {M M' : Ideal α} (hM : M.IsMaximal) (hM' : M'.IsMaximal)
    (hne : M ≠ M') : M ⊔ M' = ⊤ := by
  /-
    α : Type u
    inst✝ : Semiring α
    M M' : Ideal α
    hM : M.IsMaximal
    hM' : M'.IsMaximal
    hne : Ne M M'
    ⊢ Eq (Max.max M M') Top.top
  -/
  contrapose! hne with h
  /-
    α : Type u
    inst✝ : Semiring α
    M M' : Ideal α
    hM : M.IsMaximal
    hM' : M'.IsMaximal
    h : Ne (Max.max M M') Top.top
    ⊢ Eq M M'
  -/
  exact hM.eq_of_le hM'.ne_top (le_sup_left.trans_eq (hM'.eq_of_le h le_sup_right).symm)
  /-
    🎉 no goals
  -/


/-- **Krull's theorem**: if `I` is an ideal that is not the whole ring, then it is included in some
    maximal ideal. -/
theorem exists_le_maximal (I : Ideal α) (hI : I ≠ ⊤) : ∃ M : Ideal α, M.IsMaximal ∧ I ≤ M :=
  let ⟨m, hm⟩ := (eq_top_or_exists_le_coatom I).resolve_left hI
  ⟨m, ⟨⟨hm.1⟩, hm.2⟩⟩


/-- Krull's theorem: a nontrivial ring has a maximal ideal. -/
theorem exists_maximal [Nontrivial α] : ∃ M : Ideal α, M.IsMaximal :=
  let ⟨I, ⟨hI, _⟩⟩ := exists_le_maximal (⊥ : Ideal α) bot_ne_top
  ⟨I, hI⟩


instance [Nontrivial α] : Nontrivial (Ideal α) := by
  /-
    α : Type u
    β : Type v
    F : Type w
    inst✝¹ : Semiring α
    I : Ideal α
    a b : α
    inst✝ : Nontrivial α
    ⊢ Nontrivial (Ideal α)
  -/
  rcases@exists_maximal α _ _ with ⟨M, hM, _⟩
  /-
    case intro.mk.intro
    α : Type u
    β : Type v
    F : Type w
    inst✝¹ : Semiring α
    I : Ideal α
    a b : α
    inst✝ : Nontrivial α
    M : Ideal α
    hM : Ne M Top.top
    right✝ : ∀ (b : Ideal α), LT.lt M b → Eq b Top.top
    ⊢ Nontrivial (Ideal α)
  -/
  exact nontrivial_of_ne M ⊤ hM
  /-
    🎉 no goals
  -/


/-- If P is not properly contained in any maximal ideal then it is not properly contained
  in any proper ideal -/
theorem maximal_of_no_maximal {P : Ideal α}
    (hmax : ∀ m : Ideal α, P < m → ¬IsMaximal m) (J : Ideal α) (hPJ : P < J) : J = ⊤ := by
  /-
    α : Type u
    inst✝ : Semiring α
    P : Ideal α
    hmax : ∀ (m : Ideal α), LT.lt P m → Not m.IsMaximal
    J : Ideal α
    hPJ : LT.lt P J
    ⊢ Eq J Top.top
  -/
  by_contra hnonmax
  /-
    α : Type u
    inst✝ : Semiring α
    P : Ideal α
    hmax : ∀ (m : Ideal α), LT.lt P m → Not m.IsMaximal
    J : Ideal α
    hPJ : LT.lt P J
    hnonmax : Not (Eq J Top.top)
    ⊢ False
  -/
  rcases exists_le_maximal J hnonmax with ⟨M, hM1, hM2⟩
  /-
    case intro.intro
    α : Type u
    inst✝ : Semiring α
    P : Ideal α
    hmax : ∀ (m : Ideal α), LT.lt P m → Not m.IsMaximal
    J : Ideal α
    hPJ : LT.lt P J
    hnonmax : Not (Eq J Top.top)
    M : Ideal α
    hM1 : M.IsMaximal
    hM2 : LE.le J M
    ⊢ False
  -/
  exact hmax M (lt_of_lt_of_le hPJ hM2) hM1
  /-
    🎉 no goals
  -/


theorem IsMaximal.exists_inv {I : Ideal α} (hI : I.IsMaximal) {x} (hx : x ∉ I) :
    ∃ y, ∃ i ∈ I, y * x + i = 1 := by
  /-
    α : Type u
    inst✝ : Semiring α
    I : Ideal α
    hI : I.IsMaximal
    x : α
    hx : Not (Membership.mem I x)
    ⊢ Exists fun y => Exists fun i => And (Membership.mem I i) (Eq (HAdd.hAdd (HMu …
  -/
  cases' isMaximal_iff.1 hI with H₁ H₂
  rcases mem_span_insert.1
      (H₂ (span (insert x I)) x (Set.Subset.trans (subset_insert _ _) subset_span) hx
        (subset_span (mem_insert _ _))) with
    ⟨y, z, hz, hy⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    inst✝ : Semiring α
    I : Ideal α
    hI : I.IsMaximal
    x : α
    hx : Not (Membership.mem I x)
    H₁ : Not (Membership.mem I 1)
    H₂ : ∀ (J : Ideal α) (x : α), LE.le I J → Not (Membership.mem I x) → Membershi …
    y z : α
    hz : Membership.mem (Ideal.span ↑I) z
    hy : Eq 1 (HAdd.hAdd (HMul.hMul y x) z)
    ⊢ Exists fun y => Exists fun i => And (Membership.mem I i) (Eq (HAdd.hAdd (HMu …
  -/
  refine ⟨y, z, ?_, hy.symm⟩
  /-
    case intro.intro.intro.intro
    α : Type u
    inst✝ : Semiring α
    I : Ideal α
    hI : I.IsMaximal
    x : α
    hx : Not (Membership.mem I x)
    H₁ : Not (Membership.mem I 1)
    H₂ : ∀ (J : Ideal α) (x : α), LE.le I J → Not (Membership.mem I x) → Membershi …
    y z : α
    hz : Membership.mem (Ideal.span ↑I) z
    hy : Eq 1 (HAdd.hAdd (HMul.hMul y x) z)
    ⊢ Membership.mem I z
  -/
  rwa [← span_eq I]
  /-
    🎉 no goals
  -/


theorem sInf_isPrime_of_isChain {s : Set (Ideal α)} (hs : s.Nonempty) (hs' : IsChain (· ≤ ·) s)
    (H : ∀ p ∈ s, Ideal.IsPrime p) : (sInf s).IsPrime :=
  ⟨fun e =>
    let ⟨x, hx⟩ := hs
    (H x hx).ne_top (eq_top_iff.mpr (e.symm.trans_le (sInf_le hx))),
    fun e =>
    or_iff_not_imp_left.mpr fun hx => by
      /-
        α : Type u
        inst✝ : Semiring α
        s : Set (Ideal α)
        hs : s.Nonempty
        hs' : IsChain (fun x1 x2 => LE.le x1 x2) s
        H : ∀ (p : Ideal α), Membership.mem s p → p.IsPrime
        x✝ y✝ : α
        e : Membership.mem (InfSet.sInf s) (HMul.hMul x✝ y✝)
        hx : Not (Membership.mem (InfSet.sInf s) x✝)
        ⊢ Membership.mem (InfSet.sInf s) y✝
      -/
      rw [Ideal.mem_sInf] at hx e ⊢
      /-
        α : Type u
        inst✝ : Semiring α
        s : Set (Ideal α)
        hs : s.Nonempty
        hs' : IsChain (fun x1 x2 => LE.le x1 x2) s
        H : ∀ (p : Ideal α), Membership.mem s p → p.IsPrime
        x✝ y✝ : α
        e : ∀ ⦃I : Ideal α⦄, Membership.mem s I → Membership.mem I (HMul.hMul x✝ y✝)
        hx : Not (∀ ⦃I : Ideal α⦄, Membership.mem s I → Membership.mem I x✝)
        ⊢ ∀ ⦃I : Ideal α⦄, Membership.mem s I → Membership.mem I y✝
      -/
      push_neg at hx
      /-
        α : Type u
        inst✝ : Semiring α
        s : Set (Ideal α)
        hs : s.Nonempty
        hs' : IsChain (fun x1 x2 => LE.le x1 x2) s
        H : ∀ (p : Ideal α), Membership.mem s p → p.IsPrime
        x✝ y✝ : α
        e : ∀ ⦃I : Ideal α⦄, Membership.mem s I → Membership.mem I (HMul.hMul x✝ y✝)
        hx : Exists fun ⦃I⦄ => And (Membership.mem s I) (Not (Membership.mem I x✝))
        ⊢ ∀ ⦃I : Ideal α⦄, Membership.mem s I → Membership.mem I y✝
      -/
      obtain ⟨I, hI, hI'⟩ := hx
      /-
        case intro.intro
        α : Type u
        inst✝ : Semiring α
        s : Set (Ideal α)
        hs : s.Nonempty
        hs' : IsChain (fun x1 x2 => LE.le x1 x2) s
        H : ∀ (p : Ideal α), Membership.mem s p → p.IsPrime
        x✝ y✝ : α
        e : ∀ ⦃I : Ideal α⦄, Membership.mem s I → Membership.mem I (HMul.hMul x✝ y✝)
        I : Ideal α
        hI : Membership.mem s I
        hI' : Not (Membership.mem I x✝)
        ⊢ ∀ ⦃I : Ideal α⦄, Membership.mem s I → Membership.mem I y✝
      -/
      intro J hJ
      /-
        case intro.intro
        α : Type u
        inst✝ : Semiring α
        s : Set (Ideal α)
        hs : s.Nonempty
        hs' : IsChain (fun x1 x2 => LE.le x1 x2) s
        H : ∀ (p : Ideal α), Membership.mem s p → p.IsPrime
        x✝ y✝ : α
        e : ∀ ⦃I : Ideal α⦄, Membership.mem s I → Membership.mem I (HMul.hMul x✝ y✝)
        I : Ideal α
        hI : Membership.mem s I
        hI' : Not (Membership.mem I x✝)
        J : Ideal α
        hJ : Membership.mem s J
        ⊢ Membership.mem J y✝
      -/
      cases' hs'.total hI hJ with h h
        /-
          case intro.intro.inl
          α : Type u
          inst✝ : Semiring α
          s : Set (Ideal α)
          hs : s.Nonempty
          hs' : IsChain (fun x1 x2 => LE.le x1 x2) s
          H : ∀ (p : Ideal α), Membership.mem s p → p.IsPrime
          x✝ y✝ : α
          e : ∀ ⦃I : Ideal α⦄, Membership.mem s I → Membership.mem I (HMul.hMul x✝ y✝)
          I : Ideal α
          hI : Membership.mem s I
          hI' : Not (Membership.mem I x✝)
          J : Ideal α
          hJ : Membership.mem s J
          h : LE.le I J
          ⊢ Membership.mem J y✝
        -/
      · exact h (((H I hI).mem_or_mem (e hI)).resolve_left hI')
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.inr
          α : Type u
          inst✝ : Semiring α
          s : Set (Ideal α)
          hs : s.Nonempty
          hs' : IsChain (fun x1 x2 => LE.le x1 x2) s
          H : ∀ (p : Ideal α), Membership.mem s p → p.IsPrime
          x✝ y✝ : α
          e : ∀ ⦃I : Ideal α⦄, Membership.mem s I → Membership.mem I (HMul.hMul x✝ y✝)
          I : Ideal α
          hI : Membership.mem s I
          hI' : Not (Membership.mem I x✝)
          J : Ideal α
          hJ : Membership.mem s J
          h : LE.le J I
          ⊢ Membership.mem J y✝
        -/
      · exact ((H J hJ).mem_or_mem (e hJ)).resolve_left fun x => hI' <| h x⟩
        /-
          🎉 no goals
        -/


theorem span_singleton_prime {p : α} (hp : p ≠ 0) : IsPrime (span ({p} : Set α)) ↔ Prime p := by
  /-
    α : Type u
    inst✝ : CommSemiring α
    p : α
    hp : Ne p 0
    ⊢ Iff (Ideal.span (Singleton.singleton p)).IsPrime (Prime p)
  -/
  simp [isPrime_iff, Prime, span_singleton_eq_top, hp, mem_span_singleton]
  /-
    🎉 no goals
  -/


theorem IsMaximal.isPrime {I : Ideal α} (H : I.IsMaximal) : I.IsPrime :=
  ⟨H.1.1, @fun x y hxy =>
    or_iff_not_imp_left.2 fun hx => by
      /-
        α : Type u
        inst✝ : CommSemiring α
        I : Ideal α
        H : I.IsMaximal
        x y : α
        hxy : Membership.mem I (HMul.hMul x y)
        hx : Not (Membership.mem I x)
        ⊢ Membership.mem I y
      -/
      let J : Ideal α := Submodule.span α (insert x ↑I)
      /-
        α : Type u
        inst✝ : CommSemiring α
        I : Ideal α
        H : I.IsMaximal
        x y : α
        hxy : Membership.mem I (HMul.hMul x y)
        hx : Not (Membership.mem I x)
        J : Ideal α := Submodule.span α (Insert.insert x ↑I)
        ⊢ Membership.mem I y
      -/
      have IJ : I ≤ J := Set.Subset.trans (subset_insert _ _) subset_span
      /-
        α : Type u
        inst✝ : CommSemiring α
        I : Ideal α
        H : I.IsMaximal
        x y : α
        hxy : Membership.mem I (HMul.hMul x y)
        hx : Not (Membership.mem I x)
        J : Ideal α := Submodule.span α (Insert.insert x ↑I)
        IJ : LE.le I J
        ⊢ Membership.mem I y
      -/
      have xJ : x ∈ J := Ideal.subset_span (Set.mem_insert x I)
      /-
        α : Type u
        inst✝ : CommSemiring α
        I : Ideal α
        H : I.IsMaximal
        x y : α
        hxy : Membership.mem I (HMul.hMul x y)
        hx : Not (Membership.mem I x)
        J : Ideal α := Submodule.span α (Insert.insert x ↑I)
        IJ : LE.le I J
        xJ : Membership.mem J x
        ⊢ Membership.mem I y
      -/
      cases' isMaximal_iff.1 H with _ oJ
      /-
        case intro
        α : Type u
        inst✝ : CommSemiring α
        I : Ideal α
        H : I.IsMaximal
        x y : α
        hxy : Membership.mem I (HMul.hMul x y)
        hx : Not (Membership.mem I x)
        J : Ideal α := Submodule.span α (Insert.insert x ↑I)
        IJ : LE.le I J
        xJ : Membership.mem J x
        left✝ : Not (Membership.mem I 1)
        oJ : ∀ (J : Ideal α) (x : α), LE.le I J → Not (Membership.mem I x) → Membershi …
        ⊢ Membership.mem I y
      -/
      specialize oJ J x IJ hx xJ
      /-
        case intro
        α : Type u
        inst✝ : CommSemiring α
        I : Ideal α
        H : I.IsMaximal
        x y : α
        hxy : Membership.mem I (HMul.hMul x y)
        hx : Not (Membership.mem I x)
        J : Ideal α := Submodule.span α (Insert.insert x ↑I)
        IJ : LE.le I J
        xJ : Membership.mem J x
        left✝ : Not (Membership.mem I 1)
        oJ : Membership.mem J 1
        ⊢ Membership.mem I y
      -/
      rcases Submodule.mem_span_insert.mp oJ with ⟨a, b, h, oe⟩
      /-
        case intro.intro.intro.intro
        α : Type u
        inst✝ : CommSemiring α
        I : Ideal α
        H : I.IsMaximal
        x y : α
        hxy : Membership.mem I (HMul.hMul x y)
        hx : Not (Membership.mem I x)
        J : Ideal α := Submodule.span α (Insert.insert x ↑I)
        IJ : LE.le I J
        xJ : Membership.mem J x
        left✝ : Not (Membership.mem I 1)
        oJ : Membership.mem J 1
        a b : α
        h : Membership.mem (Submodule.span α ↑I) b
        oe : Eq 1 (HAdd.hAdd (HSMul.hSMul a x) b)
        ⊢ Membership.mem I y
      -/
      obtain F : y * 1 = y * (a • x + b) := congr_arg (fun g : α => y * g) oe
      /-
        case intro.intro.intro.intro
        α : Type u
        inst✝ : CommSemiring α
        I : Ideal α
        H : I.IsMaximal
        x y : α
        hxy : Membership.mem I (HMul.hMul x y)
        hx : Not (Membership.mem I x)
        J : Ideal α := Submodule.span α (Insert.insert x ↑I)
        IJ : LE.le I J
        xJ : Membership.mem J x
        left✝ : Not (Membership.mem I 1)
        oJ : Membership.mem J 1
        a b : α
        h : Membership.mem (Submodule.span α ↑I) b
        oe : Eq 1 (HAdd.hAdd (HSMul.hSMul a x) b)
        F : Eq (HMul.hMul y 1) (HMul.hMul y (HAdd.hAdd (HSMul.hSMul a x) b))
        ⊢ Membership.mem I y
      -/
      rw [← mul_one y, F, mul_add, mul_comm, smul_eq_mul, mul_assoc]
      /-
        case intro.intro.intro.intro
        α : Type u
        inst✝ : CommSemiring α
        I : Ideal α
        H : I.IsMaximal
        x y : α
        hxy : Membership.mem I (HMul.hMul x y)
        hx : Not (Membership.mem I x)
        J : Ideal α := Submodule.span α (Insert.insert x ↑I)
        IJ : LE.le I J
        xJ : Membership.mem J x
        left✝ : Not (Membership.mem I 1)
        oJ : Membership.mem J 1
        a b : α
        h : Membership.mem (Submodule.span α ↑I) b
        oe : Eq 1 (HAdd.hAdd (HSMul.hSMul a x) b)
        F : Eq (HMul.hMul y 1) (HMul.hMul y (HAdd.hAdd (HSMul.hSMul a x) b))
        ⊢ Membership.mem I (HAdd.hAdd (HMul.hMul a (HMul.hMul x y)) (HMul.hMul y b))
      -/
      refine Submodule.add_mem I (I.mul_mem_left a hxy) (Submodule.smul_mem I y ?_)
      /-
        case intro.intro.intro.intro
        α : Type u
        inst✝ : CommSemiring α
        I : Ideal α
        H : I.IsMaximal
        x y : α
        hxy : Membership.mem I (HMul.hMul x y)
        hx : Not (Membership.mem I x)
        J : Ideal α := Submodule.span α (Insert.insert x ↑I)
        IJ : LE.le I J
        xJ : Membership.mem J x
        left✝ : Not (Membership.mem I 1)
        oJ : Membership.mem J 1
        a b : α
        h : Membership.mem (Submodule.span α ↑I) b
        oe : Eq 1 (HAdd.hAdd (HSMul.hSMul a x) b)
        F : Eq (HMul.hMul y 1) (HMul.hMul y (HAdd.hAdd (HSMul.hSMul a x) b))
        ⊢ Membership.mem I b
      -/
      rwa [Submodule.span_eq] at h⟩
      /-
        🎉 no goals
      -/

-- see Note [lower instance priority]

instance (priority := 100) IsMaximal.isPrime' (I : Ideal α) : ∀ [_H : I.IsMaximal], I.IsPrime :=
  @IsMaximal.isPrime _ _ _


theorem exists_disjoint_powers_of_span_eq_top (s : Set α) (hs : span s = ⊤) (I : Ideal α)
    (hI : I ≠ ⊤) : ∃ r ∈ s, Disjoint (I : Set α) (Submonoid.powers r) := by
  /-
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    hs : Eq (Ideal.span s) Top.top
    I : Ideal α
    hI : Ne I Top.top
    ⊢ Exists fun r => And (Membership.mem s r) (Disjoint ↑I ↑(Submonoid.powers r))
  -/
  have ⟨M, hM, le⟩ := exists_le_maximal I hI
  /-
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    hs : Eq (Ideal.span s) Top.top
    I : Ideal α
    hI : Ne I Top.top
    M : Ideal α
    hM : M.IsMaximal
    le : LE.le I M
    ⊢ Exists fun r => And (Membership.mem s r) (Disjoint ↑I ↑(Submonoid.powers r))
  -/
  have := hM.1.1
  /-
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    hs : Eq (Ideal.span s) Top.top
    I : Ideal α
    hI : Ne I Top.top
    M : Ideal α
    hM : M.IsMaximal
    le : LE.le I M
    this : Ne M Top.top
    ⊢ Exists fun r => And (Membership.mem s r) (Disjoint ↑I ↑(Submonoid.powers r))
  -/
  rw [Ne, eq_top_iff, ← hs, span_le, Set.not_subset] at this
  /-
    α : Type u
    inst✝ : CommSemiring α
    s : Set α
    hs : Eq (Ideal.span s) Top.top
    I : Ideal α
    hI : Ne I Top.top
    M : Ideal α
    hM : M.IsMaximal
    le : LE.le I M
    this : Exists fun a => And (Membership.mem s a) (Not (Membership.mem (↑M) a))
    ⊢ Exists fun r => And (Membership.mem s r) (Disjoint ↑I ↑(Submonoid.powers r))
  -/
  have ⟨a, has, haM⟩ := this
  exact ⟨a, has, Set.disjoint_left.mpr fun x hx ⟨n, hn⟩ ↦
    haM (hM.isPrime.mem_of_pow_mem _ (le <| hn ▸ hx))⟩


theorem span_singleton_lt_span_singleton [IsDomain α] {x y : α} :
    span ({x} : Set α) < span ({y} : Set α) ↔ DvdNotUnit y x := by
  rw [lt_iff_le_not_le, span_singleton_le_span_singleton, span_singleton_le_span_singleton,
    dvd_and_not_dvd_iff]


lemma isPrime_of_maximally_disjoint (I : Ideal α)
    (S : Submonoid α)
    (disjoint : Disjoint (I : Set α) S)
    (maximally_disjoint : ∀ (J : Ideal α), I < J → ¬ Disjoint (J : Set α) S) :
    I.IsPrime where
  ne_top' := by
    /-
      α : Type u
      inst✝ : CommSemiring α
      I : Ideal α
      S : Submonoid α
      disjoint : Disjoint ↑I ↑S
      maximally_disjoint : ∀ (J : Ideal α), LT.lt I J → Not (Disjoint ↑J ↑S)
      ⊢ Ne I Top.top
    -/
    rintro rfl
    /-
      α : Type u
      inst✝ : CommSemiring α
      S : Submonoid α
      disjoint : Disjoint ↑Top.top ↑S
      maximally_disjoint : ∀ (J : Ideal α), LT.lt Top.top J → Not (Disjoint ↑J ↑S)
      ⊢ False
    -/
    have : 1 ∈ (S : Set α) := S.one_mem
    /-
      α : Type u
      inst✝ : CommSemiring α
      S : Submonoid α
      disjoint : Disjoint ↑Top.top ↑S
      maximally_disjoint : ∀ (J : Ideal α), LT.lt Top.top J → Not (Disjoint ↑J ↑S)
      this : Membership.mem (↑S) 1
      ⊢ False
    -/
    aesop
    /-
      🎉 no goals
    -/
  mem_or_mem' {x y} hxy := by
    /-
      α : Type u
      inst✝ : CommSemiring α
      I : Ideal α
      S : Submonoid α
      disjoint : Disjoint ↑I ↑S
      maximally_disjoint : ∀ (J : Ideal α), LT.lt I J → Not (Disjoint ↑J ↑S)
      x y : α
      hxy : Membership.mem I (HMul.hMul x y)
      ⊢ Or (Membership.mem I x) (Membership.mem I y)
    -/
    by_contra! rid
    /-
      α : Type u
      inst✝ : CommSemiring α
      I : Ideal α
      S : Submonoid α
      disjoint : Disjoint ↑I ↑S
      maximally_disjoint : ∀ (J : Ideal α), LT.lt I J → Not (Disjoint ↑J ↑S)
      x y : α
      hxy : Membership.mem I (HMul.hMul x y)
      rid : And (Not (Membership.mem I x)) (Not (Membership.mem I y))
      ⊢ False
    -/
    have hx := maximally_disjoint (I ⊔ span {x}) (Submodule.lt_sup_iff_not_mem.mpr rid.1)
    /-
      α : Type u
      inst✝ : CommSemiring α
      I : Ideal α
      S : Submonoid α
      disjoint : Disjoint ↑I ↑S
      maximally_disjoint : ∀ (J : Ideal α), LT.lt I J → Not (Disjoint ↑J ↑S)
      x y : α
      hxy : Membership.mem I (HMul.hMul x y)
      rid : And (Not (Membership.mem I x)) (Not (Membership.mem I y))
      hx : Not (Disjoint ↑(Max.max I (Ideal.span (Singleton.singleton x))) ↑S)
      ⊢ False
    -/
    have hy := maximally_disjoint (I ⊔ span {y}) (Submodule.lt_sup_iff_not_mem.mpr rid.2)
    simp only [Set.not_disjoint_iff, mem_inter_iff, SetLike.mem_coe, Submodule.mem_sup,
      mem_span_singleton] at hx hy
    /-
      α : Type u
      inst✝ : CommSemiring α
      I : Ideal α
      S : Submonoid α
      disjoint : Disjoint ↑I ↑S
      maximally_disjoint : ∀ (J : Ideal α), LT.lt I J → Not (Disjoint ↑J ↑S)
      x y : α
      hxy : Membership.mem I (HMul.hMul x y)
      rid : And (Not (Membership.mem I x)) (Not (Membership.mem I y))
      hx : Exists fun x_1 => And (Exists fun y => And (Membership.mem I y) (Exists f …
      hy : Exists fun x => And (Exists fun y_1 => And (Membership.mem I y_1) (Exists …
      ⊢ False
    -/
    obtain ⟨s₁, ⟨i₁, hi₁, ⟨_, ⟨r₁, rfl⟩, hr₁⟩⟩, hs₁⟩ := hx
    /-
      case intro.intro.intro.intro.intro.intro.intro
      α : Type u
      inst✝ : CommSemiring α
      I : Ideal α
      S : Submonoid α
      disjoint : Disjoint ↑I ↑S
      maximally_disjoint : ∀ (J : Ideal α), LT.lt I J → Not (Disjoint ↑J ↑S)
      x y : α
      hxy : Membership.mem I (HMul.hMul x y)
      rid : And (Not (Membership.mem I x)) (Not (Membership.mem I y))
      hy : Exists fun x => And (Exists fun y_1 => And (Membership.mem I y_1) (Exists …
      s₁ : α
      hs₁ : Membership.mem S s₁
      i₁ : α
      hi₁ : Membership.mem I i₁
      r₁ : α
      hr₁ : Eq (HAdd.hAdd i₁ (HMul.hMul x r₁)) s₁
      ⊢ False
    -/
    obtain ⟨s₂, ⟨i₂, hi₂, ⟨_, ⟨r₂, rfl⟩, hr₂⟩⟩, hs₂⟩ := hy
    refine disjoint.ne_of_mem
      (I.add_mem (I.mul_mem_left (i₁ + x * r₁) hi₂) <| I.add_mem (I.mul_mem_right (y * r₂) hi₁) <|
        I.mul_mem_right (r₁ * r₂) hxy)
      (S.mul_mem hs₁ hs₂) ?_
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      α : Type u
      inst✝ : CommSemiring α
      I : Ideal α
      S : Submonoid α
      disjoint : Disjoint ↑I ↑S
      maximally_disjoint : ∀ (J : Ideal α), LT.lt I J → Not (Disjoint ↑J ↑S)
      x y : α
      hxy : Membership.mem I (HMul.hMul x y)
      rid : And (Not (Membership.mem I x)) (Not (Membership.mem I y))
      s₁ : α
      hs₁ : Membership.mem S s₁
      i₁ : α
      hi₁ : Membership.mem I i₁
      r₁ : α
      hr₁ : Eq (HAdd.hAdd i₁ (HMul.hMul x r₁)) s₁
      s₂ : α
      hs₂ : Membership.mem S s₂
      i₂ : α
      hi₂ : Membership.mem I i₂
      r₂ : α
      hr₂ : Eq (HAdd.hAdd i₂ (HMul.hMul y r₂)) s₂
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd i₁ (HMul.hMul x r₁)) i₂) (HAdd.hAdd (HMu …
    -/
    rw [← hr₁, ← hr₂]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
      α : Type u
      inst✝ : CommSemiring α
      I : Ideal α
      S : Submonoid α
      disjoint : Disjoint ↑I ↑S
      maximally_disjoint : ∀ (J : Ideal α), LT.lt I J → Not (Disjoint ↑J ↑S)
      x y : α
      hxy : Membership.mem I (HMul.hMul x y)
      rid : And (Not (Membership.mem I x)) (Not (Membership.mem I y))
      s₁ : α
      hs₁ : Membership.mem S s₁
      i₁ : α
      hi₁ : Membership.mem I i₁
      r₁ : α
      hr₁ : Eq (HAdd.hAdd i₁ (HMul.hMul x r₁)) s₁
      s₂ : α
      hs₂ : Membership.mem S s₂
      i₂ : α
      hi₂ : Membership.mem I i₂
      r₂ : α
      hr₂ : Eq (HAdd.hAdd i₂ (HMul.hMul y r₂)) s₂
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HAdd.hAdd i₁ (HMul.hMul x r₁)) i₂) (HAdd.hAdd (HMu …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem exists_le_prime_disjoint (S : Submonoid α) (disjoint : Disjoint (I : Set α) S) :
    ∃ p : Ideal α, p.IsPrime ∧ I ≤ p ∧ Disjoint (p : Set α) S := by
  have ⟨p, hIp, hp⟩ := zorn_le_nonempty₀ {p : Ideal α | Disjoint (p : Set α) S}
    (fun c hc hc' x hx ↦ ?_) I disjoint
    /-
      case refine_2
      α : Type u
      inst✝ : CommSemiring α
      I : Ideal α
      S : Submonoid α
      disjoint : Disjoint ↑I ↑S
      p : Ideal α
      hIp : LE.le I p
      hp : Maximal (fun x => Membership.mem (setOf fun p => Disjoint ↑p ↑S) x) p
      ⊢ Exists fun p => And p.IsPrime (And (LE.le I p) (Disjoint ↑p ↑S))
    -/
  · exact ⟨p, isPrime_of_maximally_disjoint _ _ hp.1 (fun _ ↦ hp.not_prop_of_gt), hIp, hp.1⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_1
    α : Type u
    inst✝ : CommSemiring α
    I : Ideal α
    S : Submonoid α
    disjoint : Disjoint ↑I ↑S
    c : Set (Ideal α)
    hc : HasSubset.Subset c (setOf fun p => Disjoint ↑p ↑S)
    hc' : IsChain (fun x1 x2 => LE.le x1 x2) c
    x : Ideal α
    hx : Membership.mem c x
    ⊢ Exists fun ub => And (Membership.mem (setOf fun p => Disjoint ↑p ↑S) ub) (∀  …
  -/
  cases isEmpty_or_nonempty c
    /-
      case refine_1.inl
      α : Type u
      inst✝ : CommSemiring α
      I : Ideal α
      S : Submonoid α
      disjoint : Disjoint ↑I ↑S
      c : Set (Ideal α)
      hc : HasSubset.Subset c (setOf fun p => Disjoint ↑p ↑S)
      hc' : IsChain (fun x1 x2 => LE.le x1 x2) c
      x : Ideal α
      hx : Membership.mem c x
      h✝ : IsEmpty ↑c
      ⊢ Exists fun ub => And (Membership.mem (setOf fun p => Disjoint ↑p ↑S) ub) (∀  …
    -/
  · exact ⟨I, disjoint, fun J hJ ↦ isEmptyElim (⟨J, hJ⟩ : c)⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_1.inr
    α : Type u
    inst✝ : CommSemiring α
    I : Ideal α
    S : Submonoid α
    disjoint : Disjoint ↑I ↑S
    c : Set (Ideal α)
    hc : HasSubset.Subset c (setOf fun p => Disjoint ↑p ↑S)
    hc' : IsChain (fun x1 x2 => LE.le x1 x2) c
    x : Ideal α
    hx : Membership.mem c x
    h✝ : Nonempty ↑c
    ⊢ Exists fun ub => And (Membership.mem (setOf fun p => Disjoint ↑p ↑S) ub) (∀  …
  -/
  refine ⟨sSup c, Set.disjoint_left.mpr fun x hx ↦ ?_, fun _ ↦ le_sSup⟩
  /-
    case refine_1.inr
    α : Type u
    inst✝ : CommSemiring α
    I : Ideal α
    S : Submonoid α
    disjoint : Disjoint ↑I ↑S
    c : Set (Ideal α)
    hc : HasSubset.Subset c (setOf fun p => Disjoint ↑p ↑S)
    hc' : IsChain (fun x1 x2 => LE.le x1 x2) c
    x✝ : Ideal α
    hx✝ : Membership.mem c x✝
    h✝ : Nonempty ↑c
    x : α
    hx : Membership.mem (↑(SupSet.sSup c)) x
    ⊢ Not (Membership.mem (↑S) x)
  -/
  have ⟨p, hp⟩ := (Submodule.mem_iSup_of_directed _ hc'.directed).mp (sSup_eq_iSup' c ▸ hx)
  /-
    case refine_1.inr
    α : Type u
    inst✝ : CommSemiring α
    I : Ideal α
    S : Submonoid α
    disjoint : Disjoint ↑I ↑S
    c : Set (Ideal α)
    hc : HasSubset.Subset c (setOf fun p => Disjoint ↑p ↑S)
    hc' : IsChain (fun x1 x2 => LE.le x1 x2) c
    x✝ : Ideal α
    hx✝ : Membership.mem c x✝
    h✝ : Nonempty ↑c
    x : α
    hx : Membership.mem (↑(SupSet.sSup c)) x
    p : Subtype fun a => Membership.mem c a
    hp : Membership.mem (↑p) x
    ⊢ Not (Membership.mem (↑S) x)
  -/
  exact Set.disjoint_left.mp (hc p.2) hp
  /-
    🎉 no goals
  -/


theorem exists_le_prime_nmem_of_isIdempotentElem (a : α) (ha : IsIdempotentElem a) (haI : a ∉ I) :
    ∃ p : Ideal α, p.IsPrime ∧ I ≤ p ∧ a ∉ p :=
  have : Disjoint (I : Set α) (Submonoid.powers a) := Set.disjoint_right.mpr <| by
    /-
      α : Type u
      inst✝ : CommSemiring α
      I : Ideal α
      a : α
      ha : IsIdempotentElem a
      haI : Not (Membership.mem I a)
      ⊢ ∀ ⦃a_1 : α⦄, Membership.mem (↑(Submonoid.powers a)) a_1 → Not (Membership.me …
    -/
    rw [ha.coe_powers]
    /-
      α : Type u
      inst✝ : CommSemiring α
      I : Ideal α
      a : α
      ha : IsIdempotentElem a
      haI : Not (Membership.mem I a)
      ⊢ ∀ ⦃a_1 : α⦄, Membership.mem (Insert.insert 1 (Singleton.singleton a)) a_1 →  …
    -/
    rintro _ (rfl|rfl)
    /-
      case inl
      α : Type u
      inst✝ : CommSemiring α
      I : Ideal α
      a : α
      ha : IsIdempotentElem a
      haI : Not (Membership.mem I a)
      ⊢ Not (Membership.mem (↑I) 1)
    -/
    exacts [I.ne_top_iff_one.mp (ne_of_mem_of_not_mem' Submodule.mem_top haI).symm, haI]
    /-
      🎉 no goals
    -/
  have ⟨p, h1, h2, h3⟩ := exists_le_prime_disjoint _ _ this
  ⟨p, h1, h2, Set.disjoint_right.mp h3 (Submonoid.mem_powers a)⟩


theorem bot_isMaximal : IsMaximal (⊥ : Ideal K) :=
                                                                /-
                                                                  K : Type u
                                                                  inst✝ : DivisionSemiring K
                                                                  h : Eq Bot.bot Top.top
                                                                  ⊢ Not (Membership.mem Top.top 1)
                                                                -/
  ⟨⟨fun h => absurd ((eq_top_iff_one (⊤ : Ideal K)).mp rfl) (by rw [← h]; simp), fun I hI =>
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
      or_iff_not_imp_left.mp (eq_bot_or_top I) (ne_of_gt hI)⟩⟩



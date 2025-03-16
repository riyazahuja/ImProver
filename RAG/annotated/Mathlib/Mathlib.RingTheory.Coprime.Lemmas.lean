theorem Int.isCoprime_iff_gcd_eq_one {m n : ℤ} : IsCoprime m n ↔ Int.gcd m n = 1 := by
  /-
    m n : Int
    ⊢ Iff (IsCoprime m n) (Eq (m.gcd n) 1)
  -/
  constructor
    /-
      case mp
      m n : Int
      ⊢ IsCoprime m n → Eq (m.gcd n) 1
    -/
  · rintro ⟨a, b, h⟩
    /-
      case mp.intro.intro
      m n a b : Int
      h : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b n)) 1
      ⊢ Eq (m.gcd n) 1
    -/
    have : 1 = m * a + n * b := by rwa [mul_comm m, mul_comm n, eq_comm]
    /-
      case mp.intro.intro
      m n a b : Int
      h : Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b n)) 1
      this : Eq 1 (HAdd.hAdd (HMul.hMul m a) (HMul.hMul n b))
      ⊢ Eq (m.gcd n) 1
    -/
    exact Nat.dvd_one.mp (Int.gcd_dvd_iff.mpr ⟨a, b, this⟩)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      m n : Int
      ⊢ Eq (m.gcd n) 1 → IsCoprime m n
    -/
  · rw [← Int.ofNat_inj, IsCoprime, Int.gcd_eq_gcd_ab, mul_comm m, mul_comm n, Nat.cast_one]
    /-
      case mpr
      m n : Int
      ⊢ Eq (HAdd.hAdd (HMul.hMul (m.gcdA n) m) (HMul.hMul (m.gcdB n) n)) 1 → Exists  …
    -/
    intro h
    /-
      case mpr
      m n : Int
      h : Eq (HAdd.hAdd (HMul.hMul (m.gcdA n) m) (HMul.hMul (m.gcdB n) n)) 1
      ⊢ Exists fun a => Exists fun b => Eq (HAdd.hAdd (HMul.hMul a m) (HMul.hMul b n …
    -/
    exact ⟨_, _, h⟩
    /-
      🎉 no goals
    -/


theorem Nat.isCoprime_iff_coprime {m n : ℕ} : IsCoprime (m : ℤ) n ↔ Nat.Coprime m n := by
  /-
    m n : Nat
    ⊢ Iff (IsCoprime ↑m ↑n) (m.Coprime n)
  -/
  rw [Int.isCoprime_iff_gcd_eq_one, Int.gcd_natCast_natCast]
  /-
    🎉 no goals
  -/


alias ⟨IsCoprime.nat_coprime, Nat.Coprime.isCoprime⟩ := Nat.isCoprime_iff_coprime


theorem Nat.Coprime.cast {R : Type*} [CommRing R] {a b : ℕ} (h : Nat.Coprime a b) :
    IsCoprime (a : R) (b : R) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    a b : Nat
    h : a.Coprime b
    ⊢ IsCoprime ↑a ↑b
  -/
  rw [← isCoprime_iff_coprime] at h
  /-
    R : Type u_1
    inst✝ : CommRing R
    a b : Nat
    h : IsCoprime ↑a ↑b
    ⊢ IsCoprime ↑a ↑b
  -/
  rw [← Int.cast_natCast a, ← Int.cast_natCast b]
  /-
    R : Type u_1
    inst✝ : CommRing R
    a b : Nat
    h : IsCoprime ↑a ↑b
    ⊢ IsCoprime ↑↑a ↑↑b
  -/
  exact IsCoprime.intCast h
  /-
    🎉 no goals
  -/


theorem ne_zero_or_ne_zero_of_nat_coprime {A : Type u} [CommRing A] [Nontrivial A] {a b : ℕ}
    (h : Nat.Coprime a b) : (a : A) ≠ 0 ∨ (b : A) ≠ 0 :=
  IsCoprime.ne_zero_or_ne_zero (R := A) <| by
    /-
      A : Type u
      inst✝¹ : CommRing A
      inst✝ : Nontrivial A
      a b : Nat
      h : a.Coprime b
      ⊢ IsCoprime ↑a ↑b
    -/
    simpa only [map_natCast] using IsCoprime.map (Nat.Coprime.isCoprime h) (Int.castRingHom A)
    /-
      🎉 no goals
    -/


theorem IsCoprime.prod_left : (∀ i ∈ t, IsCoprime (s i) x) → IsCoprime (∏ i ∈ t, s i) x := by
  classical
  refine Finset.induction_on t (fun _ ↦ isCoprime_one_left) fun b t hbt ih H ↦ ?_
  rw [Finset.prod_insert hbt]
  rw [Finset.forall_mem_insert] at H
  exact H.1.mul_left (ih H.2)


theorem IsCoprime.prod_right : (∀ i ∈ t, IsCoprime x (s i)) → IsCoprime x (∏ i ∈ t, s i) := by
  /-
    R : Type u
    I : Type v
    inst✝ : CommSemiring R
    x : R
    s : I → R
    t : Finset I
    ⊢ (∀ (i : I), Membership.mem t i → IsCoprime x (s i)) → IsCoprime x (t.prod fu …
  -/
  simpa only [isCoprime_comm] using IsCoprime.prod_left (R := R)
  /-
    🎉 no goals
  -/


theorem IsCoprime.prod_left_iff : IsCoprime (∏ i ∈ t, s i) x ↔ ∀ i ∈ t, IsCoprime (s i) x := by
  classical
  refine Finset.induction_on t (iff_of_true isCoprime_one_left fun _ ↦ by simp) fun b t hbt ih ↦ ?_
  rw [Finset.prod_insert hbt, IsCoprime.mul_left_iff, ih, Finset.forall_mem_insert]


theorem IsCoprime.prod_right_iff : IsCoprime x (∏ i ∈ t, s i) ↔ ∀ i ∈ t, IsCoprime x (s i) := by
  /-
    R : Type u
    I : Type v
    inst✝ : CommSemiring R
    x : R
    s : I → R
    t : Finset I
    ⊢ Iff (IsCoprime x (t.prod fun i => s i)) (∀ (i : I), Membership.mem t i → IsC …
  -/
  simpa only [isCoprime_comm] using IsCoprime.prod_left_iff (R := R)
  /-
    🎉 no goals
  -/


theorem IsCoprime.of_prod_left (H1 : IsCoprime (∏ i ∈ t, s i) x) (i : I) (hit : i ∈ t) :
    IsCoprime (s i) x :=
  IsCoprime.prod_left_iff.1 H1 i hit


theorem IsCoprime.of_prod_right (H1 : IsCoprime x (∏ i ∈ t, s i)) (i : I) (hit : i ∈ t) :
    IsCoprime x (s i) :=
  IsCoprime.prod_right_iff.1 H1 i hit

-- Porting note: removed names of things due to linter, but they seem helpful

theorem Finset.prod_dvd_of_coprime :
    (t : Set I).Pairwise (IsCoprime on s) → (∀ i ∈ t, s i ∣ z) → (∏ x ∈ t, s x) ∣ z := by
  classical
  exact Finset.induction_on t (fun _ _ ↦ one_dvd z)
    (by
      intro a r har ih Hs Hs1
      rw [Finset.prod_insert har]
      have aux1 : a ∈ (↑(insert a r) : Set I) := Finset.mem_insert_self a r
      refine
        (IsCoprime.prod_right fun i hir ↦
              Hs aux1 (Finset.mem_insert_of_mem hir) <| by
                rintro rfl
                exact har hir).mul_dvd
          (Hs1 a aux1) (ih (Hs.mono ?_) fun i hi ↦ Hs1 i <| Finset.mem_insert_of_mem hi)
      simp only [Finset.coe_insert, Set.subset_insert])


theorem Fintype.prod_dvd_of_coprime [Fintype I] (Hs : Pairwise (IsCoprime on s))
    (Hs1 : ∀ i, s i ∣ z) : (∏ x, s x) ∣ z :=
  Finset.prod_dvd_of_coprime (Hs.set_pairwise _) fun i _ ↦ Hs1 i


theorem exists_sum_eq_one_iff_pairwise_coprime [DecidableEq I] (h : t.Nonempty) :
    (∃ μ : I → R, (∑ i ∈ t, μ i * ∏ j ∈ t \ {i}, s j) = 1) ↔
      Pairwise (IsCoprime on fun i : t ↦ s i) := by
  induction h using Finset.Nonempty.cons_induction with
  | singleton =>
    simp [exists_apply_eq, Pairwise, Function.onFun]
  | cons a t hat h ih =>
    rw [pairwise_cons']
    have mem : ∀ x ∈ t, a ∈ insert a t \ {x} := fun x hx ↦ by
      rw [mem_sdiff, mem_singleton]
      exact ⟨mem_insert_self _ _, fun ha ↦ hat (ha ▸ hx)⟩
    constructor
    · rintro ⟨μ, hμ⟩
      rw [sum_cons, cons_eq_insert, sdiff_singleton_eq_erase, erase_insert hat] at hμ
      refine ⟨ih.mp ⟨Pi.single h.choose (μ a * s h.choose) + μ * fun _ ↦ s a, ?_⟩, fun b hb ↦ ?_⟩
      · rw [prod_eq_mul_prod_diff_singleton h.choose_spec, ← mul_assoc, ←
          @if_pos _ _ h.choose_spec R (_ * _) 0, ← sum_pi_single', ← sum_add_distrib] at hμ
        rw [← hμ, sum_congr rfl]
        intro x hx
        dsimp -- Porting note: terms were showing as sort of `HAdd.hadd` instead of `+`
        -- this whole proof pretty much breaks and has to be rewritten from scratch
        rw [add_mul]
        congr 1
        · by_cases hx : x = h.choose
          · rw [hx, Pi.single_eq_same, Pi.single_eq_same]
          · rw [Pi.single_eq_of_ne hx, Pi.single_eq_of_ne hx, zero_mul]
        · rw [mul_assoc]
          congr
          rw [prod_eq_prod_diff_singleton_mul (mem x hx) _, mul_comm]
          congr 2
          rw [sdiff_sdiff_comm, sdiff_singleton_eq_erase a, erase_insert hat]
      · have : IsCoprime (s b) (s a) :=
          ⟨μ a * ∏ i ∈ t \ {b}, s i, ∑ i ∈ t, μ i * ∏ j ∈ t \ {i}, s j, ?_⟩
        · exact ⟨this.symm, this⟩
        rw [mul_assoc, ← prod_eq_prod_diff_singleton_mul hb, sum_mul, ← hμ, sum_congr rfl]
        intro x hx
        rw [mul_assoc]
        congr
        rw [prod_eq_prod_diff_singleton_mul (mem x hx) _]
        congr 2
        rw [sdiff_sdiff_comm, sdiff_singleton_eq_erase a, erase_insert hat]
    · rintro ⟨hs, Hb⟩
      obtain ⟨μ, hμ⟩ := ih.mpr hs
      obtain ⟨u, v, huv⟩ := IsCoprime.prod_left fun b hb ↦ (Hb b hb).right
      use fun i ↦ if i = a then u else v * μ i
      have hμ' : (∑ i ∈ t, v * ((μ i * ∏ j ∈ t \ {i}, s j) * s a)) = v * s a := by
        rw [← mul_sum, ← sum_mul, hμ, one_mul]
      rw [sum_cons, cons_eq_insert, sdiff_singleton_eq_erase, erase_insert hat]
      simp only [↓reduceIte, ite_mul]
      rw [← huv, ← hμ', sum_congr rfl]
      intro x hx
      rw [mul_assoc, if_neg fun ha : x = a ↦ hat (ha.casesOn hx)]
      rw [mul_assoc]
      congr
      rw [prod_eq_prod_diff_singleton_mul (mem x hx) _]
      congr 2
      rw [sdiff_sdiff_comm, sdiff_singleton_eq_erase a, erase_insert hat]


theorem exists_sum_eq_one_iff_pairwise_coprime' [Fintype I] [Nonempty I] [DecidableEq I] :
    (∃ μ : I → R, (∑ i : I, μ i * ∏ j ∈ {i}ᶜ, s j) = 1) ↔ Pairwise (IsCoprime on s) := by
  /-
    R : Type u
    I : Type v
    inst✝³ : CommSemiring R
    s : I → R
    inst✝² : Fintype I
    inst✝¹ : Nonempty I
    inst✝ : DecidableEq I
    ⊢ Iff (Exists fun μ => Eq (Finset.univ.sum fun i => HMul.hMul (μ i) ((HasCompl …
  -/
  convert exists_sum_eq_one_iff_pairwise_coprime Finset.univ_nonempty (s := s) using 1
  /-
    case h.e'_2.a
    R : Type u
    I : Type v
    inst✝³ : CommSemiring R
    s : I → R
    inst✝² : Fintype I
    inst✝¹ : Nonempty I
    inst✝ : DecidableEq I
    ⊢ Iff (Pairwise (Function.onFun IsCoprime s)) (Pairwise (Function.onFun IsCopr …
  -/
  simp only [Function.onFun, pairwise_subtype_iff_pairwise_finset', coe_univ, Set.pairwise_univ]
  /-
    🎉 no goals
  -/

-- Porting note: a lot of the capitalization wasn't working

theorem pairwise_coprime_iff_coprime_prod [DecidableEq I] :
    Pairwise (IsCoprime on fun i : t ↦ s i) ↔ ∀ i ∈ t, IsCoprime (s i) (∏ j ∈ t \ {i}, s j) := by
  /-
    R : Type u
    I : Type v
    inst✝¹ : CommSemiring R
    s : I → R
    t : Finset I
    inst✝ : DecidableEq I
    ⊢ Iff (Pairwise (Function.onFun IsCoprime fun i => s ↑i)) (∀ (i : I), Membersh …
  -/
  refine ⟨fun hp i hi ↦ IsCoprime.prod_right_iff.mpr fun j hj ↦ ?_, fun hp ↦ ?_⟩
    /-
      case refine_1
      R : Type u
      I : Type v
      inst✝¹ : CommSemiring R
      s : I → R
      t : Finset I
      inst✝ : DecidableEq I
      hp : Pairwise (Function.onFun IsCoprime fun i => s ↑i)
      i : I
      hi : Membership.mem t i
      j : I
      hj : Membership.mem (SDiff.sdiff t (Singleton.singleton i)) j
      ⊢ IsCoprime (s i) (s j)
    -/
  · rw [Finset.mem_sdiff, Finset.mem_singleton] at hj
    /-
      case refine_1
      R : Type u
      I : Type v
      inst✝¹ : CommSemiring R
      s : I → R
      t : Finset I
      inst✝ : DecidableEq I
      hp : Pairwise (Function.onFun IsCoprime fun i => s ↑i)
      i : I
      hi : Membership.mem t i
      j : I
      hj : And (Membership.mem t j) (Not (Eq j i))
      ⊢ IsCoprime (s i) (s j)
    -/
    obtain ⟨hj, ji⟩ := hj
    /-
      case refine_1.intro
      R : Type u
      I : Type v
      inst✝¹ : CommSemiring R
      s : I → R
      t : Finset I
      inst✝ : DecidableEq I
      hp : Pairwise (Function.onFun IsCoprime fun i => s ↑i)
      i : I
      hi : Membership.mem t i
      j : I
      hj : Membership.mem t j
      ji : Not (Eq j i)
      ⊢ IsCoprime (s i) (s j)
    -/
    refine @hp ⟨i, hi⟩ ⟨j, hj⟩ fun h ↦ ji (congrArg Subtype.val h).symm
    /-
      🎉 no goals
    -/
    -- Porting note: is there a better way compared to the old `congr_arg coe h`?
    /-
      case refine_2
      R : Type u
      I : Type v
      inst✝¹ : CommSemiring R
      s : I → R
      t : Finset I
      inst✝ : DecidableEq I
      hp : ∀ (i : I), Membership.mem t i → IsCoprime (s i) ((SDiff.sdiff t (Singleto …
      ⊢ Pairwise (Function.onFun IsCoprime fun i => s ↑i)
    -/
  · rintro ⟨i, hi⟩ ⟨j, hj⟩ h
    /-
      case refine_2.mk.mk
      R : Type u
      I : Type v
      inst✝¹ : CommSemiring R
      s : I → R
      t : Finset I
      inst✝ : DecidableEq I
      hp : ∀ (i : I), Membership.mem t i → IsCoprime (s i) ((SDiff.sdiff t (Singleto …
      i : I
      hi : Membership.mem t i
      j : I
      hj : Membership.mem t j
      h : Ne ⟨i, hi⟩ ⟨j, hj⟩
      ⊢ Function.onFun IsCoprime (fun i => s ↑i) ⟨i, hi⟩ ⟨j, hj⟩
    -/
    apply IsCoprime.prod_right_iff.mp (hp i hi)
    /-
      case refine_2.mk.mk.a
      R : Type u
      I : Type v
      inst✝¹ : CommSemiring R
      s : I → R
      t : Finset I
      inst✝ : DecidableEq I
      hp : ∀ (i : I), Membership.mem t i → IsCoprime (s i) ((SDiff.sdiff t (Singleto …
      i : I
      hi : Membership.mem t i
      j : I
      hj : Membership.mem t j
      h : Ne ⟨i, hi⟩ ⟨j, hj⟩
      ⊢ Membership.mem (SDiff.sdiff t (Singleton.singleton i)) ↑⟨j, hj⟩
    -/
    exact Finset.mem_sdiff.mpr ⟨hj, fun f ↦ h <| Subtype.ext (Finset.mem_singleton.mp f).symm⟩
    /-
      🎉 no goals
    -/


theorem IsCoprime.pow_left (H : IsCoprime x y) : IsCoprime (x ^ m) y := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y : R
    m : Nat
    H : IsCoprime x y
    ⊢ IsCoprime (HPow.hPow x m) y
  -/
  rw [← Finset.card_range m, ← Finset.prod_const]
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y : R
    m : Nat
    H : IsCoprime x y
    ⊢ IsCoprime ((Finset.range m).prod fun _x => x) y
  -/
  exact IsCoprime.prod_left fun _ _ ↦ H
  /-
    🎉 no goals
  -/


theorem IsCoprime.pow_right (H : IsCoprime x y) : IsCoprime x (y ^ n) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y : R
    n : Nat
    H : IsCoprime x y
    ⊢ IsCoprime x (HPow.hPow y n)
  -/
  rw [← Finset.card_range n, ← Finset.prod_const]
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y : R
    n : Nat
    H : IsCoprime x y
    ⊢ IsCoprime x ((Finset.range n).prod fun _x => y)
  -/
  exact IsCoprime.prod_right fun _ _ ↦ H
  /-
    🎉 no goals
  -/


theorem IsCoprime.pow (H : IsCoprime x y) : IsCoprime (x ^ m) (y ^ n) :=
  H.pow_left.pow_right


theorem IsCoprime.pow_left_iff (hm : 0 < m) : IsCoprime (x ^ m) y ↔ IsCoprime x y := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y : R
    m : Nat
    hm : LT.lt 0 m
    ⊢ Iff (IsCoprime (HPow.hPow x m) y) (IsCoprime x y)
  -/
  refine ⟨fun h ↦ ?_, IsCoprime.pow_left⟩
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y : R
    m : Nat
    hm : LT.lt 0 m
    h : IsCoprime (HPow.hPow x m) y
    ⊢ IsCoprime x y
  -/
  rw [← Finset.card_range m, ← Finset.prod_const] at h
  /-
    R : Type u
    inst✝ : CommSemiring R
    x y : R
    m : Nat
    hm : LT.lt 0 m
    h : IsCoprime ((Finset.range m).prod fun _x => x) y
    ⊢ IsCoprime x y
  -/
  exact h.of_prod_left 0 (Finset.mem_range.mpr hm)
  /-
    🎉 no goals
  -/


theorem IsCoprime.pow_right_iff (hm : 0 < m) : IsCoprime x (y ^ m) ↔ IsCoprime x y :=
  isCoprime_comm.trans <| (IsCoprime.pow_left_iff hm).trans <| isCoprime_comm


theorem IsCoprime.pow_iff (hm : 0 < m) (hn : 0 < n) : IsCoprime (x ^ m) (y ^ n) ↔ IsCoprime x y :=
  (IsCoprime.pow_left_iff hm).trans <| IsCoprime.pow_right_iff hn


theorem IsRelPrime.prod_left : (∀ i ∈ t, IsRelPrime (s i) x) → IsRelPrime (∏ i ∈ t, s i) x := by
  classical
  refine Finset.induction_on t (fun _ ↦ isRelPrime_one_left) fun b t hbt ih H ↦ ?_
  rw [Finset.prod_insert hbt]
  rw [Finset.forall_mem_insert] at H
  exact H.1.mul_left (ih H.2)


theorem IsRelPrime.prod_right : (∀ i ∈ t, IsRelPrime x (s i)) → IsRelPrime x (∏ i ∈ t, s i) := by
  /-
    α : Type u_2
    I : Type u_1
    inst✝¹ : CommMonoid α
    inst✝ : DecompositionMonoid α
    x : α
    s : I → α
    t : Finset I
    ⊢ (∀ (i : I), Membership.mem t i → IsRelPrime x (s i)) → IsRelPrime x (t.prod  …
  -/
  simpa only [isRelPrime_comm] using IsRelPrime.prod_left (α := α)
  /-
    🎉 no goals
  -/


theorem IsRelPrime.prod_left_iff : IsRelPrime (∏ i ∈ t, s i) x ↔ ∀ i ∈ t, IsRelPrime (s i) x := by
  classical
  refine Finset.induction_on t (iff_of_true isRelPrime_one_left fun _ ↦ by simp) fun b t hbt ih ↦ ?_
  rw [Finset.prod_insert hbt, IsRelPrime.mul_left_iff, ih, Finset.forall_mem_insert]


theorem IsRelPrime.prod_right_iff : IsRelPrime x (∏ i ∈ t, s i) ↔ ∀ i ∈ t, IsRelPrime x (s i) := by
  /-
    α : Type u_1
    I : Type u_2
    inst✝¹ : CommMonoid α
    inst✝ : DecompositionMonoid α
    x : α
    s : I → α
    t : Finset I
    ⊢ Iff (IsRelPrime x (t.prod fun i => s i)) (∀ (i : I), Membership.mem t i → Is …
  -/
  simpa only [isRelPrime_comm] using IsRelPrime.prod_left_iff (α := α)
  /-
    🎉 no goals
  -/


theorem IsRelPrime.of_prod_left (H1 : IsRelPrime (∏ i ∈ t, s i) x) (i : I) (hit : i ∈ t) :
    IsRelPrime (s i) x :=
  IsRelPrime.prod_left_iff.1 H1 i hit


theorem IsRelPrime.of_prod_right (H1 : IsRelPrime x (∏ i ∈ t, s i)) (i : I) (hit : i ∈ t) :
    IsRelPrime x (s i) :=
  IsRelPrime.prod_right_iff.1 H1 i hit


theorem Finset.prod_dvd_of_isRelPrime :
    (t : Set I).Pairwise (IsRelPrime on s) → (∀ i ∈ t, s i ∣ z) → (∏ x ∈ t, s x) ∣ z := by
  classical
  exact Finset.induction_on t (fun _ _ ↦ one_dvd z)
    (by
      intro a r har ih Hs Hs1
      rw [Finset.prod_insert har]
      have aux1 : a ∈ (↑(insert a r) : Set I) := Finset.mem_insert_self a r
      refine
        (IsRelPrime.prod_right fun i hir ↦
              Hs aux1 (Finset.mem_insert_of_mem hir) <| by
                rintro rfl
                exact har hir).mul_dvd
          (Hs1 a aux1) (ih (Hs.mono ?_) fun i hi ↦ Hs1 i <| Finset.mem_insert_of_mem hi)
      simp only [Finset.coe_insert, Set.subset_insert])


theorem Fintype.prod_dvd_of_isRelPrime [Fintype I] (Hs : Pairwise (IsRelPrime on s))
    (Hs1 : ∀ i, s i ∣ z) : (∏ x, s x) ∣ z :=
  Finset.prod_dvd_of_isRelPrime (Hs.set_pairwise _) fun i _ ↦ Hs1 i


theorem pairwise_isRelPrime_iff_isRelPrime_prod [DecidableEq I] :
    Pairwise (IsRelPrime on fun i : t ↦ s i) ↔ ∀ i ∈ t, IsRelPrime (s i) (∏ j ∈ t \ {i}, s j) := by
  /-
    α : Type u_2
    I : Type u_1
    inst✝² : CommMonoid α
    inst✝¹ : DecompositionMonoid α
    s : I → α
    t : Finset I
    inst✝ : DecidableEq I
    ⊢ Iff (Pairwise (Function.onFun IsRelPrime fun i => s ↑i)) (∀ (i : I), Members …
  -/
  refine ⟨fun hp i hi ↦ IsRelPrime.prod_right_iff.mpr fun j hj ↦ ?_, fun hp ↦ ?_⟩
    /-
      case refine_1
      α : Type u_2
      I : Type u_1
      inst✝² : CommMonoid α
      inst✝¹ : DecompositionMonoid α
      s : I → α
      t : Finset I
      inst✝ : DecidableEq I
      hp : Pairwise (Function.onFun IsRelPrime fun i => s ↑i)
      i : I
      hi : Membership.mem t i
      j : I
      hj : Membership.mem (SDiff.sdiff t (Singleton.singleton i)) j
      ⊢ IsRelPrime (s i) (s j)
    -/
  · rw [Finset.mem_sdiff, Finset.mem_singleton] at hj
    /-
      case refine_1
      α : Type u_2
      I : Type u_1
      inst✝² : CommMonoid α
      inst✝¹ : DecompositionMonoid α
      s : I → α
      t : Finset I
      inst✝ : DecidableEq I
      hp : Pairwise (Function.onFun IsRelPrime fun i => s ↑i)
      i : I
      hi : Membership.mem t i
      j : I
      hj : And (Membership.mem t j) (Not (Eq j i))
      ⊢ IsRelPrime (s i) (s j)
    -/
    obtain ⟨hj, ji⟩ := hj
    /-
      case refine_1.intro
      α : Type u_2
      I : Type u_1
      inst✝² : CommMonoid α
      inst✝¹ : DecompositionMonoid α
      s : I → α
      t : Finset I
      inst✝ : DecidableEq I
      hp : Pairwise (Function.onFun IsRelPrime fun i => s ↑i)
      i : I
      hi : Membership.mem t i
      j : I
      hj : Membership.mem t j
      ji : Not (Eq j i)
      ⊢ IsRelPrime (s i) (s j)
    -/
    exact @hp ⟨i, hi⟩ ⟨j, hj⟩ fun h ↦ ji (congrArg Subtype.val h).symm
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      I : Type u_1
      inst✝² : CommMonoid α
      inst✝¹ : DecompositionMonoid α
      s : I → α
      t : Finset I
      inst✝ : DecidableEq I
      hp : ∀ (i : I), Membership.mem t i → IsRelPrime (s i) ((SDiff.sdiff t (Singlet …
      ⊢ Pairwise (Function.onFun IsRelPrime fun i => s ↑i)
    -/
  · rintro ⟨i, hi⟩ ⟨j, hj⟩ h
    /-
      case refine_2.mk.mk
      α : Type u_2
      I : Type u_1
      inst✝² : CommMonoid α
      inst✝¹ : DecompositionMonoid α
      s : I → α
      t : Finset I
      inst✝ : DecidableEq I
      hp : ∀ (i : I), Membership.mem t i → IsRelPrime (s i) ((SDiff.sdiff t (Singlet …
      i : I
      hi : Membership.mem t i
      j : I
      hj : Membership.mem t j
      h : Ne ⟨i, hi⟩ ⟨j, hj⟩
      ⊢ Function.onFun IsRelPrime (fun i => s ↑i) ⟨i, hi⟩ ⟨j, hj⟩
    -/
    apply IsRelPrime.prod_right_iff.mp (hp i hi)
    /-
      case refine_2.mk.mk.a
      α : Type u_2
      I : Type u_1
      inst✝² : CommMonoid α
      inst✝¹ : DecompositionMonoid α
      s : I → α
      t : Finset I
      inst✝ : DecidableEq I
      hp : ∀ (i : I), Membership.mem t i → IsRelPrime (s i) ((SDiff.sdiff t (Singlet …
      i : I
      hi : Membership.mem t i
      j : I
      hj : Membership.mem t j
      h : Ne ⟨i, hi⟩ ⟨j, hj⟩
      ⊢ Membership.mem (SDiff.sdiff t (Singleton.singleton i)) ↑⟨j, hj⟩
    -/
    exact Finset.mem_sdiff.mpr ⟨hj, fun f ↦ h <| Subtype.ext (Finset.mem_singleton.mp f).symm⟩
    /-
      🎉 no goals
    -/


theorem pow_left (H : IsRelPrime x y) : IsRelPrime (x ^ m) y := by
  /-
    α : Type u_1
    inst✝¹ : CommMonoid α
    inst✝ : DecompositionMonoid α
    x y : α
    m : Nat
    H : IsRelPrime x y
    ⊢ IsRelPrime (HPow.hPow x m) y
  -/
  rw [← Finset.card_range m, ← Finset.prod_const]
  /-
    α : Type u_1
    inst✝¹ : CommMonoid α
    inst✝ : DecompositionMonoid α
    x y : α
    m : Nat
    H : IsRelPrime x y
    ⊢ IsRelPrime ((Finset.range m).prod fun _x => x) y
  -/
  exact IsRelPrime.prod_left fun _ _ ↦ H
  /-
    🎉 no goals
  -/


theorem pow_right (H : IsRelPrime x y) : IsRelPrime x (y ^ n) := by
  /-
    α : Type u_1
    inst✝¹ : CommMonoid α
    inst✝ : DecompositionMonoid α
    x y : α
    n : Nat
    H : IsRelPrime x y
    ⊢ IsRelPrime x (HPow.hPow y n)
  -/
  rw [← Finset.card_range n, ← Finset.prod_const]
  /-
    α : Type u_1
    inst✝¹ : CommMonoid α
    inst✝ : DecompositionMonoid α
    x y : α
    n : Nat
    H : IsRelPrime x y
    ⊢ IsRelPrime x ((Finset.range n).prod fun _x => y)
  -/
  exact IsRelPrime.prod_right fun _ _ ↦ H
  /-
    🎉 no goals
  -/


theorem pow (H : IsRelPrime x y) : IsRelPrime (x ^ m) (y ^ n) :=
  H.pow_left.pow_right


theorem pow_left_iff (hm : 0 < m) : IsRelPrime (x ^ m) y ↔ IsRelPrime x y := by
  /-
    α : Type u_1
    inst✝¹ : CommMonoid α
    inst✝ : DecompositionMonoid α
    x y : α
    m : Nat
    hm : LT.lt 0 m
    ⊢ Iff (IsRelPrime (HPow.hPow x m) y) (IsRelPrime x y)
  -/
  refine ⟨fun h ↦ ?_, IsRelPrime.pow_left⟩
  /-
    α : Type u_1
    inst✝¹ : CommMonoid α
    inst✝ : DecompositionMonoid α
    x y : α
    m : Nat
    hm : LT.lt 0 m
    h : IsRelPrime (HPow.hPow x m) y
    ⊢ IsRelPrime x y
  -/
  rw [← Finset.card_range m, ← Finset.prod_const] at h
  /-
    α : Type u_1
    inst✝¹ : CommMonoid α
    inst✝ : DecompositionMonoid α
    x y : α
    m : Nat
    hm : LT.lt 0 m
    h : IsRelPrime ((Finset.range m).prod fun _x => x) y
    ⊢ IsRelPrime x y
  -/
  exact h.of_prod_left 0 (Finset.mem_range.mpr hm)
  /-
    🎉 no goals
  -/


theorem pow_right_iff (hm : 0 < m) : IsRelPrime x (y ^ m) ↔ IsRelPrime x y :=
  isRelPrime_comm.trans <| (IsRelPrime.pow_left_iff hm).trans <| isRelPrime_comm


theorem pow_iff (hm : 0 < m) (hn : 0 < n) :
    IsRelPrime (x ^ m) (y ^ n) ↔ IsRelPrime x y :=
  (IsRelPrime.pow_left_iff hm).trans (IsRelPrime.pow_right_iff hn)



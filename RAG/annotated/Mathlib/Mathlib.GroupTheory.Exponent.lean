/-- A predicate on a monoid saying that there is a positive integer `n` such that `g ^ n = 1`
  for all `g`. -/
@[to_additive
      "A predicate on an additive monoid saying that there is a positive integer `n` such\n
      that `n • g = 0` for all `g`."]
def ExponentExists :=
  ∃ n, 0 < n ∧ ∀ g : G, g ^ n = 1


/-- The exponent of a group is the smallest positive integer `n` such that `g ^ n = 1` for all
  `g ∈ G` if it exists, otherwise it is zero by convention. -/
@[to_additive
      "The exponent of an additive group is the smallest positive integer `n` such that\n
      `n • g = 0` for all `g ∈ G` if it exists, otherwise it is zero by convention."]
noncomputable def exponent :=
  if h : ExponentExists G then Nat.find h else 0


@[simp]
theorem _root_.AddMonoid.exponent_additive :
    AddMonoid.exponent (Additive G) = exponent G := rfl


@[simp]
theorem exponent_multiplicative {G : Type*} [AddMonoid G] :
    exponent (Multiplicative G) = AddMonoid.exponent G := rfl


open MulOpposite in
@[to_additive (attr := simp)]
theorem _root_.MulOpposite.exponent : exponent (MulOpposite G) = exponent G := by
  /-
    G : Type u
    inst✝ : Monoid G
    ⊢ Eq (Monoid.exponent (MulOpposite G)) (Monoid.exponent G)
  -/
  simp only [Monoid.exponent, ExponentExists]
  /-
    G : Type u
    inst✝ : Monoid G
    ⊢ Eq (dite (Exists fun n => And (LT.lt 0 n) (∀ (g : MulOpposite G), Eq (HPow.h …
  -/
  congr!
  /-
    case h₂.h.e'_1.h.h.e'_2.a
    G : Type u
    inst✝ : Monoid G
    h_congr_thm✝ : ∀ {b c : Prop} {α : Type} {x : Decidable b} [inst : Decidable c …
    h✝ : Exists fun n => And (LT.lt 0 n) (∀ (g : G), Eq (HPow.hPow g n) 1)
    x✝ : Nat
    ⊢ Iff (∀ (g : MulOpposite G), Eq (HPow.hPow g x✝) 1) (∀ (g : G), Eq (HPow.hPow …
  -/
  all_goals exact ⟨(op_injective <| · <| op ·), (unop_injective <| · <| unop ·)⟩
  /-
    🎉 no goals
  -/


@[to_additive]
theorem ExponentExists.isOfFinOrder (h : ExponentExists G) {g : G} : IsOfFinOrder g :=
                                        /-
                                          G : Type u
                                          inst✝ : Monoid G
                                          h : Monoid.ExponentExists G
                                          g : G
                                          ⊢ Exists fun n => And (LT.lt 0 n) (Eq (HPow.hPow g n) 1)
                                        -/
  isOfFinOrder_iff_pow_eq_one.mpr <| by peel 2 h; exact this g
                                                  /-
                                                    🎉 no goals
                                                  -/


@[to_additive]
theorem ExponentExists.orderOf_pos (h : ExponentExists G) (g : G) : 0 < orderOf g :=
  h.isOfFinOrder.orderOf_pos


@[to_additive]
theorem exponent_ne_zero : exponent G ≠ 0 ↔ ExponentExists G := by
  /-
    G : Type u
    inst✝ : Monoid G
    ⊢ Iff (Ne (Monoid.exponent G) 0) (Monoid.ExponentExists G)
  -/
  rw [exponent]
  /-
    G : Type u
    inst✝ : Monoid G
    ⊢ Iff (Ne (dite (Monoid.ExponentExists G) (fun h => Nat.find h) fun h => 0) 0) …
  -/
  split_ifs with h
    /-
      case pos
      G : Type u
      inst✝ : Monoid G
      h : Monoid.ExponentExists G
      ⊢ Iff (Ne (Nat.find h) 0) (Monoid.ExponentExists G)
    -/
  · simp [h, @not_lt_zero' ℕ]
    /-
      🎉 no goals
    -/
  --if this isn't done this way, `to_additive` freaks
    /-
      case neg
      G : Type u
      inst✝ : Monoid G
      h : Not (Monoid.ExponentExists G)
      ⊢ Iff (Ne 0 0) (Monoid.ExponentExists G)
    -/
  · tauto
    /-
      🎉 no goals
    -/


@[to_additive]
protected alias ⟨_, ExponentExists.exponent_ne_zero⟩ := exponent_ne_zero


@[to_additive]
theorem exponent_pos : 0 < exponent G ↔ ExponentExists G :=
  pos_iff_ne_zero.trans exponent_ne_zero


@[to_additive]
protected alias ⟨_, ExponentExists.exponent_pos⟩ := exponent_pos


@[to_additive]
theorem exponent_eq_zero_iff : exponent G = 0 ↔ ¬ExponentExists G :=
  exponent_ne_zero.not_right


@[to_additive exponent_eq_zero_addOrder_zero]
theorem exponent_eq_zero_of_order_zero {g : G} (hg : orderOf g = 0) : exponent G = 0 :=
  exponent_eq_zero_iff.mpr fun h ↦ h.orderOf_pos g |>.ne' hg


/-- The exponent is zero iff for all nonzero `n`, one can find a `g` such that `g ^ n ≠ 1`. -/
@[to_additive "The exponent is zero iff for all nonzero `n`, one can find a `g` such that
`n • g ≠ 0`."]
theorem exponent_eq_zero_iff_forall : exponent G = 0 ↔ ∀ n > 0, ∃ g : G, g ^ n ≠ 1 := by
  /-
    G : Type u
    inst✝ : Monoid G
    ⊢ Iff (Eq (Monoid.exponent G) 0) (∀ (n : Nat), GT.gt n 0 → Exists fun g => Ne  …
  -/
  rw [exponent_eq_zero_iff, ExponentExists]
  /-
    G : Type u
    inst✝ : Monoid G
    ⊢ Iff (Not (Exists fun n => And (LT.lt 0 n) (∀ (g : G), Eq (HPow.hPow g n) 1)) …
  -/
  push_neg
  /-
    G : Type u
    inst✝ : Monoid G
    ⊢ Iff (∀ (n : Nat), LT.lt 0 n → Exists fun g => Ne (HPow.hPow g n) 1) (∀ (n :  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive exponent_nsmul_eq_zero]
theorem pow_exponent_eq_one (g : G) : g ^ exponent G = 1 := by
  /-
    G : Type u
    inst✝ : Monoid G
    g : G
    ⊢ Eq (HPow.hPow g (Monoid.exponent G)) 1
  -/
  by_cases h : ExponentExists G
    /-
      case pos
      G : Type u
      inst✝ : Monoid G
      g : G
      h : Monoid.ExponentExists G
      ⊢ Eq (HPow.hPow g (Monoid.exponent G)) 1
    -/
  · simp_rw [exponent, dif_pos h]
    /-
      case pos
      G : Type u
      inst✝ : Monoid G
      g : G
      h : Monoid.ExponentExists G
      ⊢ Eq (HPow.hPow g (Nat.find h)) 1
    -/
    exact (Nat.find_spec h).2 g
    /-
      🎉 no goals
    -/
    /-
      case neg
      G : Type u
      inst✝ : Monoid G
      g : G
      h : Not (Monoid.ExponentExists G)
      ⊢ Eq (HPow.hPow g (Monoid.exponent G)) 1
    -/
  · simp_rw [exponent, dif_neg h, pow_zero]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem pow_eq_mod_exponent {n : ℕ} (g : G) : g ^ n = g ^ (n % exponent G) :=
  calc
                                                                       /-
                                                                         G : Type u
                                                                         inst✝ : Monoid G
                                                                         n : Nat
                                                                         g : G
                                                                         ⊢ Eq (HPow.hPow g n) (HPow.hPow g (HAdd.hAdd (HMod.hMod n (Monoid.exponent G)) …
                                                                       -/
    g ^ n = g ^ (n % exponent G + exponent G * (n / exponent G)) := by rw [Nat.mod_add_div]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                   /-
                                     G : Type u
                                     inst✝ : Monoid G
                                     n : Nat
                                     g : G
                                     ⊢ Eq (HPow.hPow g (HAdd.hAdd (HMod.hMod n (Monoid.exponent G)) (HMul.hMul (Mon …
                                   -/
    _ = g ^ (n % exponent G) := by simp [pow_add, pow_mul, pow_exponent_eq_one]
                                   /-
                                     🎉 no goals
                                   -/


@[to_additive]
theorem exponent_pos_of_exists (n : ℕ) (hpos : 0 < n) (hG : ∀ g : G, g ^ n = 1) :
    0 < exponent G :=
  ExponentExists.exponent_pos ⟨n, hpos, hG⟩


@[to_additive]
theorem exponent_min' (n : ℕ) (hpos : 0 < n) (hG : ∀ g : G, g ^ n = 1) : exponent G ≤ n := by
  /-
    G : Type u
    inst✝ : Monoid G
    n : Nat
    hpos : LT.lt 0 n
    hG : ∀ (g : G), Eq (HPow.hPow g n) 1
    ⊢ LE.le (Monoid.exponent G) n
  -/
  rw [exponent, dif_pos]
    /-
      G : Type u
      inst✝ : Monoid G
      n : Nat
      hpos : LT.lt 0 n
      hG : ∀ (g : G), Eq (HPow.hPow g n) 1
      ⊢ LE.le (Nat.find ?hc) n
    -/
  · apply Nat.find_min'
    /-
      case h
      G : Type u
      inst✝ : Monoid G
      n : Nat
      hpos : LT.lt 0 n
      hG : ∀ (g : G), Eq (HPow.hPow g n) 1
      ⊢ And (LT.lt 0 n) (∀ (g : G), Eq (HPow.hPow g n) 1)
    -/
    exact ⟨hpos, hG⟩
    /-
      🎉 no goals
    -/
    /-
      case hc
      G : Type u
      inst✝ : Monoid G
      n : Nat
      hpos : LT.lt 0 n
      hG : ∀ (g : G), Eq (HPow.hPow g n) 1
      ⊢ Monoid.ExponentExists G
    -/
  · exact ⟨n, hpos, hG⟩
    /-
      🎉 no goals
    -/


@[to_additive]
theorem exponent_min (m : ℕ) (hpos : 0 < m) (hm : m < exponent G) : ∃ g : G, g ^ m ≠ 1 := by
  /-
    G : Type u
    inst✝ : Monoid G
    m : Nat
    hpos : LT.lt 0 m
    hm : LT.lt m (Monoid.exponent G)
    ⊢ Exists fun g => Ne (HPow.hPow g m) 1
  -/
  by_contra! h
  /-
    G : Type u
    inst✝ : Monoid G
    m : Nat
    hpos : LT.lt 0 m
    hm : LT.lt m (Monoid.exponent G)
    h : ∀ (g : G), Eq (HPow.hPow g m) 1
    ⊢ False
  -/
  have hcon : exponent G ≤ m := exponent_min' m hpos h
  /-
    G : Type u
    inst✝ : Monoid G
    m : Nat
    hpos : LT.lt 0 m
    hm : LT.lt m (Monoid.exponent G)
    h : ∀ (g : G), Eq (HPow.hPow g m) 1
    hcon : LE.le (Monoid.exponent G) m
    ⊢ False
  -/
  omega
  /-
    🎉 no goals
  -/


@[to_additive AddMonoid.exp_eq_one_iff]
theorem exp_eq_one_iff : exponent G = 1 ↔ Subsingleton G := by
  /-
    G : Type u
    inst✝ : Monoid G
    ⊢ Iff (Eq (Monoid.exponent G) 1) (Subsingleton G)
  -/
  refine ⟨fun eq_one => ⟨fun a b => ?a_eq_b⟩, fun h => le_antisymm ?le ?ge⟩
    /-
      case a_eq_b
      G : Type u
      inst✝ : Monoid G
      eq_one : Eq (Monoid.exponent G) 1
      a b : G
      ⊢ Eq a b
    -/
  · rw [← pow_one a, ← pow_one b, ← eq_one, Monoid.pow_exponent_eq_one, Monoid.pow_exponent_eq_one]
    /-
      🎉 no goals
    -/
    /-
      case le
      G : Type u
      inst✝ : Monoid G
      h : Subsingleton G
      ⊢ LE.le (Monoid.exponent G) 1
    -/
  · apply exponent_min' _ Nat.one_pos
    /-
      case le
      G : Type u
      inst✝ : Monoid G
      h : Subsingleton G
      ⊢ ∀ (g : G), Eq (HPow.hPow g 1) 1
    -/
    simp [eq_iff_true_of_subsingleton]
    /-
      🎉 no goals
    -/
    /-
      case ge
      G : Type u
      inst✝ : Monoid G
      h : Subsingleton G
      ⊢ LE.le 1 (Monoid.exponent G)
    -/
  · apply Nat.succ_le_of_lt
    /-
      case ge.h
      G : Type u
      inst✝ : Monoid G
      h : Subsingleton G
      ⊢ LT.lt 0 (Monoid.exponent G)
    -/
    apply exponent_pos_of_exists 1 Nat.one_pos
    /-
      case ge.h
      G : Type u
      inst✝ : Monoid G
      h : Subsingleton G
      ⊢ ∀ (g : G), Eq (HPow.hPow g 1) 1
    -/
    simp [eq_iff_true_of_subsingleton]
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp) AddMonoid.exp_eq_one_of_subsingleton]
theorem exp_eq_one_of_subsingleton [hs : Subsingleton G] : exponent G = 1 :=
  exp_eq_one_iff.mpr hs


@[to_additive addOrder_dvd_exponent]
theorem order_dvd_exponent (g : G) : orderOf g ∣ exponent G :=
  orderOf_dvd_of_pow_eq_one <| pow_exponent_eq_one g


@[to_additive]
theorem orderOf_le_exponent (h : ExponentExists G) (g : G) : orderOf g ≤ exponent G :=
  Nat.le_of_dvd h.exponent_pos (order_dvd_exponent g)


@[to_additive]
theorem exponent_dvd_iff_forall_pow_eq_one {n : ℕ} : exponent G ∣ n ↔ ∀ g : G, g ^ n = 1 := by
  /-
    G : Type u
    inst✝ : Monoid G
    n : Nat
    ⊢ Iff (Dvd.dvd (Monoid.exponent G) n) (∀ (g : G), Eq (HPow.hPow g n) 1)
  -/
  rcases n.eq_zero_or_pos with (rfl | hpos)
    /-
      case inl
      G : Type u
      inst✝ : Monoid G
      ⊢ Iff (Dvd.dvd (Monoid.exponent G) 0) (∀ (g : G), Eq (HPow.hPow g 0) 1)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    G : Type u
    inst✝ : Monoid G
    n : Nat
    hpos : GT.gt n 0
    ⊢ Iff (Dvd.dvd (Monoid.exponent G) n) (∀ (g : G), Eq (HPow.hPow g n) 1)
  -/
  constructor
    /-
      case inr.mp
      G : Type u
      inst✝ : Monoid G
      n : Nat
      hpos : GT.gt n 0
      ⊢ Dvd.dvd (Monoid.exponent G) n → ∀ (g : G), Eq (HPow.hPow g n) 1
    -/
  · intro h g
    /-
      case inr.mp
      G : Type u
      inst✝ : Monoid G
      n : Nat
      hpos : GT.gt n 0
      h : Dvd.dvd (Monoid.exponent G) n
      g : G
      ⊢ Eq (HPow.hPow g n) 1
    -/
    rw [Nat.dvd_iff_mod_eq_zero] at h
    /-
      case inr.mp
      G : Type u
      inst✝ : Monoid G
      n : Nat
      hpos : GT.gt n 0
      h : Eq (HMod.hMod n (Monoid.exponent G)) 0
      g : G
      ⊢ Eq (HPow.hPow g n) 1
    -/
    rw [pow_eq_mod_exponent, h, pow_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr.mpr
      G : Type u
      inst✝ : Monoid G
      n : Nat
      hpos : GT.gt n 0
      ⊢ (∀ (g : G), Eq (HPow.hPow g n) 1) → Dvd.dvd (Monoid.exponent G) n
    -/
  · intro hG
    /-
      case inr.mpr
      G : Type u
      inst✝ : Monoid G
      n : Nat
      hpos : GT.gt n 0
      hG : ∀ (g : G), Eq (HPow.hPow g n) 1
      ⊢ Dvd.dvd (Monoid.exponent G) n
    -/
    by_contra h
    /-
      case inr.mpr
      G : Type u
      inst✝ : Monoid G
      n : Nat
      hpos : GT.gt n 0
      hG : ∀ (g : G), Eq (HPow.hPow g n) 1
      h : Not (Dvd.dvd (Monoid.exponent G) n)
      ⊢ False
    -/
    rw [Nat.dvd_iff_mod_eq_zero, ← Ne, ← pos_iff_ne_zero] at h
    /-
      case inr.mpr
      G : Type u
      inst✝ : Monoid G
      n : Nat
      hpos : GT.gt n 0
      hG : ∀ (g : G), Eq (HPow.hPow g n) 1
      h : LT.lt 0 (HMod.hMod n (Monoid.exponent G))
      ⊢ False
    -/
    have h₂ : n % exponent G < exponent G := Nat.mod_lt _ (exponent_pos_of_exists n hpos hG)
    have h₃ : exponent G ≤ n % exponent G := by
      apply exponent_min' _ h
      simp_rw [← pow_eq_mod_exponent]
      exact hG
    /-
      case inr.mpr
      G : Type u
      inst✝ : Monoid G
      n : Nat
      hpos : GT.gt n 0
      hG : ∀ (g : G), Eq (HPow.hPow g n) 1
      h : LT.lt 0 (HMod.hMod n (Monoid.exponent G))
      h₂ : LT.lt (HMod.hMod n (Monoid.exponent G)) (Monoid.exponent G)
      h₃ : LE.le (Monoid.exponent G) (HMod.hMod n (Monoid.exponent G))
      ⊢ False
    -/
    exact h₂.not_le h₃
    /-
      🎉 no goals
    -/


@[to_additive]
alias ⟨_, exponent_dvd_of_forall_pow_eq_one⟩ := exponent_dvd_iff_forall_pow_eq_one


@[to_additive]
theorem exponent_dvd {n : ℕ} : exponent G ∣ n ↔ ∀ g : G, orderOf g ∣ n := by
  /-
    G : Type u
    inst✝ : Monoid G
    n : Nat
    ⊢ Iff (Dvd.dvd (Monoid.exponent G) n) (∀ (g : G), Dvd.dvd (orderOf g) n)
  -/
  simp_rw [exponent_dvd_iff_forall_pow_eq_one, orderOf_dvd_iff_pow_eq_one]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem lcm_orderOf_dvd_exponent [Fintype G] :
    (Finset.univ : Finset G).lcm orderOf ∣ exponent G := by
  /-
    G : Type u
    inst✝¹ : Monoid G
    inst✝ : Fintype G
    ⊢ Dvd.dvd (Finset.univ.lcm orderOf) (Monoid.exponent G)
  -/
  apply Finset.lcm_dvd
  /-
    case a
    G : Type u
    inst✝¹ : Monoid G
    inst✝ : Fintype G
    ⊢ ∀ (b : G), Membership.mem Finset.univ b → Dvd.dvd (orderOf b) (Monoid.expone …
  -/
  intro g _
  /-
    case a
    G : Type u
    inst✝¹ : Monoid G
    inst✝ : Fintype G
    g : G
    a✝ : Membership.mem Finset.univ g
    ⊢ Dvd.dvd (orderOf g) (Monoid.exponent G)
  -/
  exact order_dvd_exponent g
  /-
    🎉 no goals
  -/


@[to_additive exists_addOrderOf_eq_pow_padic_val_nat_add_exponent]
theorem _root_.Nat.Prime.exists_orderOf_eq_pow_factorization_exponent {p : ℕ} (hp : p.Prime) :
    ∃ g : G, orderOf g = p ^ (exponent G).factorization p := by
  /-
    G : Type u
    inst✝ : Monoid G
    p : Nat
    hp : Nat.Prime p
    ⊢ Exists fun g => Eq (orderOf g) (HPow.hPow p ((Monoid.exponent G).factorizati …
  -/
  haveI := Fact.mk hp
  /-
    G : Type u
    inst✝ : Monoid G
    p : Nat
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    ⊢ Exists fun g => Eq (orderOf g) (HPow.hPow p ((Monoid.exponent G).factorizati …
  -/
  rcases eq_or_ne ((exponent G).factorization p) 0 with (h | h)
    /-
      case inl
      G : Type u
      inst✝ : Monoid G
      p : Nat
      hp : Nat.Prime p
      this : Fact (Nat.Prime p)
      h : Eq ((Monoid.exponent G).factorization p) 0
      ⊢ Exists fun g => Eq (orderOf g) (HPow.hPow p ((Monoid.exponent G).factorizati …
    -/
  · refine ⟨1, by rw [h, pow_zero, orderOf_one]⟩
    /-
      🎉 no goals
    -/
  have he : 0 < exponent G :=
    Ne.bot_lt fun ht => by
      rw [ht] at h
      apply h
      rw [bot_eq_zero, Nat.factorization_zero, Finsupp.zero_apply]
  /-
    case inr
    G : Type u
    inst✝ : Monoid G
    p : Nat
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    h : Ne ((Monoid.exponent G).factorization p) 0
    he : LT.lt 0 (Monoid.exponent G)
    ⊢ Exists fun g => Eq (orderOf g) (HPow.hPow p ((Monoid.exponent G).factorizati …
  -/
  rw [← Finsupp.mem_support_iff] at h
  obtain ⟨g, hg⟩ : ∃ g : G, g ^ (exponent G / p) ≠ 1 := by
    suffices key : ¬exponent G ∣ exponent G / p by
      rwa [exponent_dvd_iff_forall_pow_eq_one, not_forall] at key
    exact fun hd =>
      hp.one_lt.not_le
        ((mul_le_iff_le_one_left he).mp <|
          Nat.le_of_dvd he <| Nat.mul_dvd_of_dvd_div (Nat.dvd_of_mem_primeFactors h) hd)
  /-
    case inr.intro
    G : Type u
    inst✝ : Monoid G
    p : Nat
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    h : Membership.mem (Monoid.exponent G).factorization.support p
    he : LT.lt 0 (Monoid.exponent G)
    g : G
    hg : Ne (HPow.hPow g (HDiv.hDiv (Monoid.exponent G) p)) 1
    ⊢ Exists fun g => Eq (orderOf g) (HPow.hPow p ((Monoid.exponent G).factorizati …
  -/
  obtain ⟨k, hk : exponent G = p ^ _ * k⟩ := Nat.ordProj_dvd _ _
  /-
    case inr.intro.intro
    G : Type u
    inst✝ : Monoid G
    p : Nat
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    h : Membership.mem (Monoid.exponent G).factorization.support p
    he : LT.lt 0 (Monoid.exponent G)
    g : G
    hg : Ne (HPow.hPow g (HDiv.hDiv (Monoid.exponent G) p)) 1
    k : Nat
    hk : Eq (Monoid.exponent G) (HMul.hMul (HPow.hPow p ((Monoid.exponent G).facto …
    ⊢ Exists fun g => Eq (orderOf g) (HPow.hPow p ((Monoid.exponent G).factorizati …
  -/
  obtain ⟨t, ht⟩ := Nat.exists_eq_succ_of_ne_zero (Finsupp.mem_support_iff.mp h)
  /-
    case inr.intro.intro.intro
    G : Type u
    inst✝ : Monoid G
    p : Nat
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    h : Membership.mem (Monoid.exponent G).factorization.support p
    he : LT.lt 0 (Monoid.exponent G)
    g : G
    hg : Ne (HPow.hPow g (HDiv.hDiv (Monoid.exponent G) p)) 1
    k : Nat
    hk : Eq (Monoid.exponent G) (HMul.hMul (HPow.hPow p ((Monoid.exponent G).facto …
    t : Nat
    ht : Eq ((Monoid.exponent G).factorization p) t.succ
    ⊢ Exists fun g => Eq (orderOf g) (HPow.hPow p ((Monoid.exponent G).factorizati …
  -/
  refine ⟨g ^ k, ?_⟩
  /-
    case inr.intro.intro.intro
    G : Type u
    inst✝ : Monoid G
    p : Nat
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    h : Membership.mem (Monoid.exponent G).factorization.support p
    he : LT.lt 0 (Monoid.exponent G)
    g : G
    hg : Ne (HPow.hPow g (HDiv.hDiv (Monoid.exponent G) p)) 1
    k : Nat
    hk : Eq (Monoid.exponent G) (HMul.hMul (HPow.hPow p ((Monoid.exponent G).facto …
    t : Nat
    ht : Eq ((Monoid.exponent G).factorization p) t.succ
    ⊢ Eq (orderOf (HPow.hPow g k)) (HPow.hPow p ((Monoid.exponent G).factorization …
  -/
  rw [ht]
  /-
    case inr.intro.intro.intro
    G : Type u
    inst✝ : Monoid G
    p : Nat
    hp : Nat.Prime p
    this : Fact (Nat.Prime p)
    h : Membership.mem (Monoid.exponent G).factorization.support p
    he : LT.lt 0 (Monoid.exponent G)
    g : G
    hg : Ne (HPow.hPow g (HDiv.hDiv (Monoid.exponent G) p)) 1
    k : Nat
    hk : Eq (Monoid.exponent G) (HMul.hMul (HPow.hPow p ((Monoid.exponent G).facto …
    t : Nat
    ht : Eq ((Monoid.exponent G).factorization p) t.succ
    ⊢ Eq (orderOf (HPow.hPow g k)) (HPow.hPow p t.succ)
  -/
  apply orderOf_eq_prime_pow
    /-
      case inr.intro.intro.intro.hnot
      G : Type u
      inst✝ : Monoid G
      p : Nat
      hp : Nat.Prime p
      this : Fact (Nat.Prime p)
      h : Membership.mem (Monoid.exponent G).factorization.support p
      he : LT.lt 0 (Monoid.exponent G)
      g : G
      hg : Ne (HPow.hPow g (HDiv.hDiv (Monoid.exponent G) p)) 1
      k : Nat
      hk : Eq (Monoid.exponent G) (HMul.hMul (HPow.hPow p ((Monoid.exponent G).facto …
      t : Nat
      ht : Eq ((Monoid.exponent G).factorization p) t.succ
      ⊢ Not (Eq (HPow.hPow (HPow.hPow g k) (HPow.hPow p t)) 1)
    -/
  · rwa [hk, mul_comm, ht, pow_succ, ← mul_assoc, Nat.mul_div_cancel _ hp.pos, pow_mul] at hg
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro.intro.hfin
      G : Type u
      inst✝ : Monoid G
      p : Nat
      hp : Nat.Prime p
      this : Fact (Nat.Prime p)
      h : Membership.mem (Monoid.exponent G).factorization.support p
      he : LT.lt 0 (Monoid.exponent G)
      g : G
      hg : Ne (HPow.hPow g (HDiv.hDiv (Monoid.exponent G) p)) 1
      k : Nat
      hk : Eq (Monoid.exponent G) (HMul.hMul (HPow.hPow p ((Monoid.exponent G).facto …
      t : Nat
      ht : Eq ((Monoid.exponent G).factorization p) t.succ
      ⊢ Eq (HPow.hPow (HPow.hPow g k) (HPow.hPow p (HAdd.hAdd t 1))) 1
    -/
  · rw [← Nat.succ_eq_add_one, ← ht, ← pow_mul, mul_comm, ← hk]
    /-
      case inr.intro.intro.intro.hfin
      G : Type u
      inst✝ : Monoid G
      p : Nat
      hp : Nat.Prime p
      this : Fact (Nat.Prime p)
      h : Membership.mem (Monoid.exponent G).factorization.support p
      he : LT.lt 0 (Monoid.exponent G)
      g : G
      hg : Ne (HPow.hPow g (HDiv.hDiv (Monoid.exponent G) p)) 1
      k : Nat
      hk : Eq (Monoid.exponent G) (HMul.hMul (HPow.hPow p ((Monoid.exponent G).facto …
      t : Nat
      ht : Eq ((Monoid.exponent G).factorization p) t.succ
      ⊢ Eq (HPow.hPow g (Monoid.exponent G)) 1
    -/
    exact pow_exponent_eq_one g
    /-
      🎉 no goals
    -/


variable {G} in
open Nat in
/-- If two commuting elements `x` and `y` of a monoid have order `n` and `m`, there is an element
of order `lcm n m`. The result actually gives an explicit (computable) element, written as the
product of a power of `x` and a power of `y`. See also the result below if you don't need the
explicit formula. -/
@[to_additive "If two commuting elements `x` and `y` of an additive monoid have order `n` and `m`,
there is an element of order `lcm n m`. The result actually gives an explicit (computable) element,
written as the sum of a multiple of `x` and a multiple of `y`. See also the result below if you
don't need the explicit formula."]
lemma _root_.Commute.orderOf_mul_pow_eq_lcm {x y : G} (h : Commute x y) (hx : orderOf x ≠ 0)
    (hy : orderOf y ≠ 0) :
    orderOf (x ^ (orderOf x / (factorizationLCMLeft (orderOf x) (orderOf y))) *
      y ^ (orderOf y / factorizationLCMRight (orderOf x) (orderOf y))) =
      Nat.lcm (orderOf x) (orderOf y) := by
  /-
    G : Type u
    inst✝ : Monoid G
    x y : G
    h : Commute x y
    hx : Ne (orderOf x) 0
    hy : Ne (orderOf y) 0
    ⊢ Eq (orderOf (HMul.hMul (HPow.hPow x (HDiv.hDiv (orderOf x) ((orderOf x).fact …
  -/
  rw [(h.pow_pow _ _).orderOf_mul_eq_mul_orderOf_of_coprime]
  /-
    G : Type u
    inst✝ : Monoid G
    x y : G
    h : Commute x y
    hx : Ne (orderOf x) 0
    hy : Ne (orderOf y) 0
    ⊢ Eq (HMul.hMul (orderOf (HPow.hPow x (HDiv.hDiv (orderOf x) ((orderOf x).fact …
  -/
  all_goals iterate 2 rw [orderOf_pow_orderOf_div]; try rw [Coprime]
  all_goals simp [factorizationLCMLeft_mul_factorizationLCMRight, factorizationLCMLeft_dvd_left,
    factorizationLCMRight_dvd_right, coprime_factorizationLCMLeft_factorizationLCMRight, hx, hy]


open Submonoid in
/-- If two commuting elements `x` and `y` of a monoid have order `n` and `m`, then there is an
element of order `lcm n m` that lies in the subgroup generated by `x` and `y`. -/
@[to_additive "If two commuting elements `x` and `y` of an additive monoid have order `n` and `m`,
then there is an element of order `lcm n m` that lies in the additive subgroup generated by `x`
and `y`."]
theorem _root_.Commute.exists_orderOf_eq_lcm {x y : G} (h : Commute x y) :
    ∃ z ∈ closure {x, y}, orderOf z = Nat.lcm (orderOf x) (orderOf y) := by
  /-
    G : Type u
    inst✝ : Monoid G
    x y : G
    h : Commute x y
    ⊢ Exists fun z => And (Membership.mem (Submonoid.closure (Insert.insert x (Sin …
  -/
  by_cases hx : orderOf x = 0 <;> by_cases hy : orderOf y = 0
    /-
      case pos
      G : Type u
      inst✝ : Monoid G
      x y : G
      h : Commute x y
      hx : Eq (orderOf x) 0
      hy : Eq (orderOf y) 0
      ⊢ Exists fun z => And (Membership.mem (Submonoid.closure (Insert.insert x (Sin …
    -/
  · exact ⟨x, subset_closure (by simp), by simp [hx]⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      G : Type u
      inst✝ : Monoid G
      x y : G
      h : Commute x y
      hx : Eq (orderOf x) 0
      hy : Not (Eq (orderOf y) 0)
      ⊢ Exists fun z => And (Membership.mem (Submonoid.closure (Insert.insert x (Sin …
    -/
  · exact ⟨x, subset_closure (by simp), by simp [hx]⟩
    /-
      🎉 no goals
    -/
    /-
      case pos
      G : Type u
      inst✝ : Monoid G
      x y : G
      h : Commute x y
      hx : Not (Eq (orderOf x) 0)
      hy : Eq (orderOf y) 0
      ⊢ Exists fun z => And (Membership.mem (Submonoid.closure (Insert.insert x (Sin …
    -/
  · exact ⟨y, subset_closure (by simp), by simp [hy]⟩
    /-
      🎉 no goals
    -/
  · exact ⟨_, mul_mem (pow_mem (subset_closure (by simp)) _) (pow_mem (subset_closure (by simp)) _),
      h.orderOf_mul_pow_eq_lcm hx hy⟩


/-- A nontrivial monoid has prime exponent `p` if and only if every non-identity element has
order `p`. -/
@[to_additive]
lemma exponent_eq_prime_iff {G : Type*} [Monoid G] [Nontrivial G] {p : ℕ} (hp : p.Prime) :
    Monoid.exponent G = p ↔ ∀ g : G, g ≠ 1 → orderOf g = p := by
  /-
    G : Type u_1
    inst✝¹ : Monoid G
    inst✝ : Nontrivial G
    p : Nat
    hp : Nat.Prime p
    ⊢ Iff (Eq (Monoid.exponent G) p) (∀ (g : G), Ne g 1 → Eq (orderOf g) p)
  -/
  refine ⟨fun hG g hg ↦ ?_, fun h ↦ dvd_antisymm ?_ ?_⟩
    /-
      case refine_1
      G : Type u_1
      inst✝¹ : Monoid G
      inst✝ : Nontrivial G
      p : Nat
      hp : Nat.Prime p
      hG : Eq (Monoid.exponent G) p
      g : G
      hg : Ne g 1
      ⊢ Eq (orderOf g) p
    -/
  · rw [Ne, ← orderOf_eq_one_iff] at hg
    /-
      case refine_1
      G : Type u_1
      inst✝¹ : Monoid G
      inst✝ : Nontrivial G
      p : Nat
      hp : Nat.Prime p
      hG : Eq (Monoid.exponent G) p
      g : G
      hg : Not (Eq (orderOf g) 1)
      ⊢ Eq (orderOf g) p
    -/
    exact Eq.symm <| (hp.dvd_iff_eq hg).mp <| hG ▸ Monoid.order_dvd_exponent g
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      inst✝¹ : Monoid G
      inst✝ : Nontrivial G
      p : Nat
      hp : Nat.Prime p
      h : ∀ (g : G), Ne g 1 → Eq (orderOf g) p
      ⊢ Dvd.dvd (Monoid.exponent G) p
    -/
  · rw [exponent_dvd]
    /-
      case refine_2
      G : Type u_1
      inst✝¹ : Monoid G
      inst✝ : Nontrivial G
      p : Nat
      hp : Nat.Prime p
      h : ∀ (g : G), Ne g 1 → Eq (orderOf g) p
      ⊢ ∀ (g : G), Dvd.dvd (orderOf g) p
    -/
    intro g
    /-
      case refine_2
      G : Type u_1
      inst✝¹ : Monoid G
      inst✝ : Nontrivial G
      p : Nat
      hp : Nat.Prime p
      h : ∀ (g : G), Ne g 1 → Eq (orderOf g) p
      g : G
      ⊢ Dvd.dvd (orderOf g) p
    -/
    by_cases hg : g = 1
      /-
        case pos
        G : Type u_1
        inst✝¹ : Monoid G
        inst✝ : Nontrivial G
        p : Nat
        hp : Nat.Prime p
        h : ∀ (g : G), Ne g 1 → Eq (orderOf g) p
        g : G
        hg : Eq g 1
        ⊢ Dvd.dvd (orderOf g) p
      -/
    · simp [hg]
      /-
        🎉 no goals
      -/
      /-
        case neg
        G : Type u_1
        inst✝¹ : Monoid G
        inst✝ : Nontrivial G
        p : Nat
        hp : Nat.Prime p
        h : ∀ (g : G), Ne g 1 → Eq (orderOf g) p
        g : G
        hg : Not (Eq g 1)
        ⊢ Dvd.dvd (orderOf g) p
      -/
    · rw [h g hg]
      /-
        🎉 no goals
      -/
    /-
      case refine_3
      G : Type u_1
      inst✝¹ : Monoid G
      inst✝ : Nontrivial G
      p : Nat
      hp : Nat.Prime p
      h : ∀ (g : G), Ne g 1 → Eq (orderOf g) p
      ⊢ Dvd.dvd p (Monoid.exponent G)
    -/
  · obtain ⟨g, hg⟩ := exists_ne (1 : G)
    /-
      case refine_3.intro
      G : Type u_1
      inst✝¹ : Monoid G
      inst✝ : Nontrivial G
      p : Nat
      hp : Nat.Prime p
      h : ∀ (g : G), Ne g 1 → Eq (orderOf g) p
      g : G
      hg : Ne g 1
      ⊢ Dvd.dvd p (Monoid.exponent G)
    -/
    simpa [h g hg] using Monoid.order_dvd_exponent g
    /-
      🎉 no goals
    -/


@[to_additive]
theorem exponent_ne_zero_iff_range_orderOf_finite (h : ∀ g : G, 0 < orderOf g) :
    exponent G ≠ 0 ↔ (Set.range (orderOf : G → ℕ)).Finite := by
  /-
    G : Type u
    inst✝ : Monoid G
    h : ∀ (g : G), LT.lt 0 (orderOf g)
    ⊢ Iff (Ne (Monoid.exponent G) 0) (Set.range orderOf).Finite
  -/
  refine ⟨fun he => ?_, fun he => ?_⟩
    /-
      case refine_1
      G : Type u
      inst✝ : Monoid G
      h : ∀ (g : G), LT.lt 0 (orderOf g)
      he : Ne (Monoid.exponent G) 0
      ⊢ (Set.range orderOf).Finite
    -/
  · by_contra h
    /-
      case refine_1
      G : Type u
      inst✝ : Monoid G
      h✝ : ∀ (g : G), LT.lt 0 (orderOf g)
      he : Ne (Monoid.exponent G) 0
      h : Not (Set.range orderOf).Finite
      ⊢ False
    -/
    obtain ⟨m, ⟨t, rfl⟩, het⟩ := Set.Infinite.exists_gt h (exponent G)
    /-
      case refine_1.intro.intro.intro
      G : Type u
      inst✝ : Monoid G
      h✝ : ∀ (g : G), LT.lt 0 (orderOf g)
      he : Ne (Monoid.exponent G) 0
      h : Not (Set.range orderOf).Finite
      t : G
      het : LT.lt (Monoid.exponent G) (orderOf t)
      ⊢ False
    -/
    exact pow_ne_one_of_lt_orderOf he het (pow_exponent_eq_one t)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u
      inst✝ : Monoid G
      h : ∀ (g : G), LT.lt 0 (orderOf g)
      he : (Set.range orderOf).Finite
      ⊢ Ne (Monoid.exponent G) 0
    -/
  · lift Set.range (orderOf (G := G)) to Finset ℕ using he with t ht
    have htpos : 0 < t.prod id := by
      refine Finset.prod_pos fun a ha => ?_
      rw [← Finset.mem_coe, ht] at ha
      obtain ⟨k, rfl⟩ := ha
      exact h k
    suffices exponent G ∣ t.prod id by
      intro h
      rw [h, zero_dvd_iff] at this
      exact htpos.ne' this
    /-
      case refine_2.intro
      G : Type u
      inst✝ : Monoid G
      h : ∀ (g : G), LT.lt 0 (orderOf g)
      t : Finset Nat
      ht : Eq (↑t) (Set.range orderOf)
      exponent_ne_zero_iff_range_orderOf_finite✝ exponent_ne_zero_iff_range_orderOf_ …
      htpos : LT.lt 0 (t.prod id)
      ⊢ Dvd.dvd (Monoid.exponent G) (t.prod id)
    -/
    rw [exponent_dvd]
    /-
      case refine_2.intro
      G : Type u
      inst✝ : Monoid G
      h : ∀ (g : G), LT.lt 0 (orderOf g)
      t : Finset Nat
      ht : Eq (↑t) (Set.range orderOf)
      exponent_ne_zero_iff_range_orderOf_finite✝ exponent_ne_zero_iff_range_orderOf_ …
      htpos : LT.lt 0 (t.prod id)
      ⊢ ∀ (g : G), Dvd.dvd (orderOf g) (t.prod id)
    -/
    intro g
    /-
      case refine_2.intro
      G : Type u
      inst✝ : Monoid G
      h : ∀ (g : G), LT.lt 0 (orderOf g)
      t : Finset Nat
      ht : Eq (↑t) (Set.range orderOf)
      exponent_ne_zero_iff_range_orderOf_finite✝ exponent_ne_zero_iff_range_orderOf_ …
      htpos : LT.lt 0 (t.prod id)
      g : G
      ⊢ Dvd.dvd (orderOf g) (t.prod id)
    -/
    apply Finset.dvd_prod_of_mem id (?_ : orderOf g ∈ _)
    /-
      G : Type u
      inst✝ : Monoid G
      h : ∀ (g : G), LT.lt 0 (orderOf g)
      t : Finset Nat
      ht : Eq (↑t) (Set.range orderOf)
      exponent_ne_zero_iff_range_orderOf_finite✝ exponent_ne_zero_iff_range_orderOf_ …
      htpos : LT.lt 0 (t.prod id)
      g : G
      ⊢ Membership.mem t (orderOf g)
    -/
    rw [← Finset.mem_coe, ht]
    /-
      G : Type u
      inst✝ : Monoid G
      h : ∀ (g : G), LT.lt 0 (orderOf g)
      t : Finset Nat
      ht : Eq (↑t) (Set.range orderOf)
      exponent_ne_zero_iff_range_orderOf_finite✝ exponent_ne_zero_iff_range_orderOf_ …
      htpos : LT.lt 0 (t.prod id)
      g : G
      ⊢ Membership.mem (Set.range orderOf) (orderOf g)
    -/
    exact Set.mem_range_self g
    /-
      🎉 no goals
    -/


@[to_additive]
theorem exponent_eq_zero_iff_range_orderOf_infinite (h : ∀ g : G, 0 < orderOf g) :
    exponent G = 0 ↔ (Set.range (orderOf : G → ℕ)).Infinite := by
  /-
    G : Type u
    inst✝ : Monoid G
    h : ∀ (g : G), LT.lt 0 (orderOf g)
    ⊢ Iff (Eq (Monoid.exponent G) 0) (Set.range orderOf).Infinite
  -/
  have := exponent_ne_zero_iff_range_orderOf_finite h
  /-
    G : Type u
    inst✝ : Monoid G
    h : ∀ (g : G), LT.lt 0 (orderOf g)
    this : Iff (Ne (Monoid.exponent G) 0) (Set.range orderOf).Finite
    ⊢ Iff (Eq (Monoid.exponent G) 0) (Set.range orderOf).Infinite
  -/
  rwa [Ne, not_iff_comm, Iff.comm] at this
  /-
    🎉 no goals
  -/


@[to_additive]
theorem lcm_orderOf_eq_exponent [Fintype G] : (Finset.univ : Finset G).lcm orderOf = exponent G :=
  Nat.dvd_antisymm
    (lcm_orderOf_dvd_exponent G)
    (exponent_dvd.mpr fun g => Finset.dvd_lcm (Finset.mem_univ g))


/--
If there exists an injective, multiplication-preserving map from `G` to `H`,
then the exponent of `G` divides the exponent of `H`.
-/
@[to_additive "If there exists an injective, addition-preserving map from `G` to `H`,
then the exponent of `G` divides the exponent of `H`."]
theorem exponent_dvd_of_monoidHom (e : G →* H) (e_inj : Function.Injective e) :
    Monoid.exponent G ∣ Monoid.exponent H :=
  exponent_dvd_of_forall_pow_eq_one fun g => e_inj (by
    /-
      G : Type u
      inst✝¹ : Monoid G
      H : Type u_1
      inst✝ : Monoid H
      e : MonoidHom G H
      e_inj : Function.Injective ⇑e
      g : G
      ⊢ Eq (e (HPow.hPow g (Monoid.exponent H))) (e 1)
    -/
    rw [map_pow, pow_exponent_eq_one, map_one])
    /-
      🎉 no goals
    -/


/--
If there exists a multiplication-preserving equivalence between `G` and `H`,
then the exponent of `G` is equal to the exponent of `H`.
-/
@[to_additive "If there exists a addition-preserving equivalence between `G` and `H`,
then the exponent of `G` is equal to the exponent of `H`."]
theorem exponent_eq_of_mulEquiv (e : G ≃* H) : Monoid.exponent G = Monoid.exponent H :=
  Nat.dvd_antisymm
    (exponent_dvd_of_monoidHom e e.injective)
    (exponent_dvd_of_monoidHom e.symm e.symm.injective)


variable (G) in
@[to_additive (attr := simp)]
theorem _root_.Submonoid.exponent_top :
    Monoid.exponent (⊤ : Submonoid G) = Monoid.exponent G :=
  exponent_eq_of_mulEquiv Submonoid.topEquiv


@[to_additive]
theorem _root_.Submonoid.pow_exponent_eq_one {S : Submonoid G} {g : G} (g_in_s : g ∈ S) :
    g ^ (Monoid.exponent S) = 1 := by
  /-
    G : Type u
    inst✝ : Monoid G
    S : Submonoid G
    g : G
    g_in_s : Membership.mem S g
    ⊢ Eq (HPow.hPow g (Monoid.exponent (Subtype fun x => Membership.mem S x))) 1
  -/
  have := Monoid.pow_exponent_eq_one (⟨g, g_in_s⟩ : S)
  /-
    G : Type u
    inst✝ : Monoid G
    S : Submonoid G
    g : G
    g_in_s : Membership.mem S g
    this : Eq (HPow.hPow ⟨g, g_in_s⟩ (Monoid.exponent (Subtype fun x => Membership …
    ⊢ Eq (HPow.hPow g (Monoid.exponent (Subtype fun x => Membership.mem S x))) 1
  -/
  rwa [SubmonoidClass.mk_pow, ← OneMemClass.coe_eq_one] at this
  /-
    🎉 no goals
  -/


@[to_additive]
theorem ExponentExists.of_finite : ExponentExists G := by
  /-
    G : Type u
    inst✝¹ : LeftCancelMonoid G
    inst✝ : Finite G
    ⊢ Monoid.ExponentExists G
  -/
  let _inst := Fintype.ofFinite G
  /-
    G : Type u
    inst✝¹ : LeftCancelMonoid G
    inst✝ : Finite G
    _inst : Fintype G := Fintype.ofFinite G
    ⊢ Monoid.ExponentExists G
  -/
  simp only [Monoid.ExponentExists]
  /-
    G : Type u
    inst✝¹ : LeftCancelMonoid G
    inst✝ : Finite G
    _inst : Fintype G := Fintype.ofFinite G
    ⊢ Exists fun n => And (LT.lt 0 n) (∀ (g : G), Eq (HPow.hPow g n) 1)
  -/
  refine ⟨(Finset.univ : Finset G).lcm orderOf, ?_, fun g => ?_⟩
    /-
      case refine_1
      G : Type u
      inst✝¹ : LeftCancelMonoid G
      inst✝ : Finite G
      _inst : Fintype G := Fintype.ofFinite G
      ⊢ LT.lt 0 (Finset.univ.lcm orderOf)
    -/
  · simpa [pos_iff_ne_zero, Finset.lcm_eq_zero_iff] using fun x => (_root_.orderOf_pos x).ne'
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u
      inst✝¹ : LeftCancelMonoid G
      inst✝ : Finite G
      _inst : Fintype G := Fintype.ofFinite G
      g : G
      ⊢ Eq (HPow.hPow g (Finset.univ.lcm orderOf)) 1
    -/
  · rw [← orderOf_dvd_iff_pow_eq_one, lcm_orderOf_eq_exponent]
    /-
      case refine_2
      G : Type u
      inst✝¹ : LeftCancelMonoid G
      inst✝ : Finite G
      _inst : Fintype G := Fintype.ofFinite G
      g : G
      ⊢ Dvd.dvd (orderOf g) (Monoid.exponent G)
    -/
    exact order_dvd_exponent g
    /-
      🎉 no goals
    -/


@[to_additive]
theorem exponent_ne_zero_of_finite : exponent G ≠ 0 :=
  ExponentExists.of_finite.exponent_ne_zero


@[to_additive AddMonoid.one_lt_exponent]
lemma one_lt_exponent [Nontrivial G] : 1 < Monoid.exponent G := by
  /-
    G : Type u
    inst✝² : LeftCancelMonoid G
    inst✝¹ : Finite G
    inst✝ : Nontrivial G
    ⊢ LT.lt 1 (Monoid.exponent G)
  -/
  rw [Nat.one_lt_iff_ne_zero_and_ne_one]
  /-
    G : Type u
    inst✝² : LeftCancelMonoid G
    inst✝¹ : Finite G
    inst✝ : Nontrivial G
    ⊢ And (Ne (Monoid.exponent G) 0) (Ne (Monoid.exponent G) 1)
  -/
  exact ⟨exponent_ne_zero_of_finite, mt exp_eq_one_iff.mp (not_subsingleton G)⟩
  /-
    🎉 no goals
  -/


@[to_additive]
instance neZero_exponent_of_finite : NeZero <| Monoid.exponent G :=
  ⟨Monoid.exponent_ne_zero_of_finite⟩


@[to_additive]
theorem exists_orderOf_eq_exponent (hG : ExponentExists G) : ∃ g : G, orderOf g = exponent G := by
  /-
    G : Type u
    inst✝ : CommMonoid G
    hG : Monoid.ExponentExists G
    ⊢ Exists fun g => Eq (orderOf g) (Monoid.exponent G)
  -/
  have he := hG.exponent_ne_zero
  /-
    G : Type u
    inst✝ : CommMonoid G
    hG : Monoid.ExponentExists G
    he : Ne (Monoid.exponent G) 0
    ⊢ Exists fun g => Eq (orderOf g) (Monoid.exponent G)
  -/
  have hne : (Set.range (orderOf : G → ℕ)).Nonempty := ⟨1, 1, orderOf_one⟩
  have hfin : (Set.range (orderOf : G → ℕ)).Finite := by
    rwa [← exponent_ne_zero_iff_range_orderOf_finite hG.orderOf_pos]
  /-
    G : Type u
    inst✝ : CommMonoid G
    hG : Monoid.ExponentExists G
    he : Ne (Monoid.exponent G) 0
    hne : (Set.range orderOf).Nonempty
    hfin : (Set.range orderOf).Finite
    ⊢ Exists fun g => Eq (orderOf g) (Monoid.exponent G)
  -/
  obtain ⟨t, ht⟩ := hne.csSup_mem hfin
  /-
    case intro
    G : Type u
    inst✝ : CommMonoid G
    hG : Monoid.ExponentExists G
    he : Ne (Monoid.exponent G) 0
    hne : (Set.range orderOf).Nonempty
    hfin : (Set.range orderOf).Finite
    t : G
    ht : Eq (orderOf t) (SupSet.sSup (Set.range orderOf))
    ⊢ Exists fun g => Eq (orderOf g) (Monoid.exponent G)
  -/
  use t
  /-
    case h
    G : Type u
    inst✝ : CommMonoid G
    hG : Monoid.ExponentExists G
    he : Ne (Monoid.exponent G) 0
    hne : (Set.range orderOf).Nonempty
    hfin : (Set.range orderOf).Finite
    t : G
    ht : Eq (orderOf t) (SupSet.sSup (Set.range orderOf))
    ⊢ Eq (orderOf t) (Monoid.exponent G)
  -/
  apply Nat.dvd_antisymm (order_dvd_exponent _)
  /-
    case h
    G : Type u
    inst✝ : CommMonoid G
    hG : Monoid.ExponentExists G
    he : Ne (Monoid.exponent G) 0
    hne : (Set.range orderOf).Nonempty
    hfin : (Set.range orderOf).Finite
    t : G
    ht : Eq (orderOf t) (SupSet.sSup (Set.range orderOf))
    ⊢ Dvd.dvd (Monoid.exponent G) (orderOf t)
  -/
  refine Nat.dvd_of_primeFactorsList_subperm he ?_
  /-
    case h
    G : Type u
    inst✝ : CommMonoid G
    hG : Monoid.ExponentExists G
    he : Ne (Monoid.exponent G) 0
    hne : (Set.range orderOf).Nonempty
    hfin : (Set.range orderOf).Finite
    t : G
    ht : Eq (orderOf t) (SupSet.sSup (Set.range orderOf))
    ⊢ (Monoid.exponent G).primeFactorsList.Subperm (orderOf t).primeFactorsList
  -/
  rw [List.subperm_ext_iff]
  /-
    case h
    G : Type u
    inst✝ : CommMonoid G
    hG : Monoid.ExponentExists G
    he : Ne (Monoid.exponent G) 0
    hne : (Set.range orderOf).Nonempty
    hfin : (Set.range orderOf).Finite
    t : G
    ht : Eq (orderOf t) (SupSet.sSup (Set.range orderOf))
    ⊢ ∀ (x : Nat), Membership.mem (Monoid.exponent G).primeFactorsList x → LE.le ( …
  -/
  by_contra! h
  /-
    case h
    G : Type u
    inst✝ : CommMonoid G
    hG : Monoid.ExponentExists G
    he : Ne (Monoid.exponent G) 0
    hne : (Set.range orderOf).Nonempty
    hfin : (Set.range orderOf).Finite
    t : G
    ht : Eq (orderOf t) (SupSet.sSup (Set.range orderOf))
    h : Exists fun x => And (Membership.mem (Monoid.exponent G).primeFactorsList x …
    ⊢ False
  -/
  obtain ⟨p, hp, hpe⟩ := h
  /-
    case h.intro.intro
    G : Type u
    inst✝ : CommMonoid G
    hG : Monoid.ExponentExists G
    he : Ne (Monoid.exponent G) 0
    hne : (Set.range orderOf).Nonempty
    hfin : (Set.range orderOf).Finite
    t : G
    ht : Eq (orderOf t) (SupSet.sSup (Set.range orderOf))
    p : Nat
    hp : Membership.mem (Monoid.exponent G).primeFactorsList p
    hpe : LT.lt (List.count p (orderOf t).primeFactorsList) (List.count p (Monoid. …
    ⊢ False
  -/
  replace hp := Nat.prime_of_mem_primeFactorsList hp
  /-
    case h.intro.intro
    G : Type u
    inst✝ : CommMonoid G
    hG : Monoid.ExponentExists G
    he : Ne (Monoid.exponent G) 0
    hne : (Set.range orderOf).Nonempty
    hfin : (Set.range orderOf).Finite
    t : G
    ht : Eq (orderOf t) (SupSet.sSup (Set.range orderOf))
    p : Nat
    hpe : LT.lt (List.count p (orderOf t).primeFactorsList) (List.count p (Monoid. …
    hp : Nat.Prime p
    ⊢ False
  -/
  simp only [Nat.primeFactorsList_count_eq] at hpe
  /-
    case h.intro.intro
    G : Type u
    inst✝ : CommMonoid G
    hG : Monoid.ExponentExists G
    he : Ne (Monoid.exponent G) 0
    hne : (Set.range orderOf).Nonempty
    hfin : (Set.range orderOf).Finite
    t : G
    ht : Eq (orderOf t) (SupSet.sSup (Set.range orderOf))
    p : Nat
    hp : Nat.Prime p
    hpe : LT.lt ((orderOf t).factorization p) ((Monoid.exponent G).factorization p)
    ⊢ False
  -/
  set k := (orderOf t).factorization p with hk
  /-
    case h.intro.intro
    G : Type u
    inst✝ : CommMonoid G
    hG : Monoid.ExponentExists G
    he : Ne (Monoid.exponent G) 0
    hne : (Set.range orderOf).Nonempty
    hfin : (Set.range orderOf).Finite
    t : G
    ht : Eq (orderOf t) (SupSet.sSup (Set.range orderOf))
    p : Nat
    hp : Nat.Prime p
    k : Nat := (orderOf t).factorization p
    hpe : LT.lt k ((Monoid.exponent G).factorization p)
    hk : Eq k ((orderOf t).factorization p)
    ⊢ False
  -/
  obtain ⟨g, hg⟩ := hp.exists_orderOf_eq_pow_factorization_exponent G
  suffices orderOf t < orderOf (t ^ p ^ k * g) by
    rw [ht] at this
    exact this.not_le (le_csSup hfin.bddAbove <| Set.mem_range_self _)
  /-
    case h.intro.intro.intro
    G : Type u
    inst✝ : CommMonoid G
    hG : Monoid.ExponentExists G
    he : Ne (Monoid.exponent G) 0
    hne : (Set.range orderOf).Nonempty
    hfin : (Set.range orderOf).Finite
    t : G
    ht : Eq (orderOf t) (SupSet.sSup (Set.range orderOf))
    p : Nat
    hp : Nat.Prime p
    k : Nat := (orderOf t).factorization p
    hpe : LT.lt k ((Monoid.exponent G).factorization p)
    hk : Eq k ((orderOf t).factorization p)
    g : G
    hg : Eq (orderOf g) (HPow.hPow p ((Monoid.exponent G).factorization p))
    ⊢ LT.lt (orderOf t) (orderOf (HMul.hMul (HPow.hPow t (HPow.hPow p k)) g))
  -/
  have hpk : p ^ k ∣ orderOf t := Nat.ordProj_dvd _ _
  have hpk' : orderOf (t ^ p ^ k) = orderOf t / p ^ k := by
    rw [orderOf_pow' t (pow_ne_zero k hp.ne_zero), Nat.gcd_eq_right hpk]
  /-
    case h.intro.intro.intro
    G : Type u
    inst✝ : CommMonoid G
    hG : Monoid.ExponentExists G
    he : Ne (Monoid.exponent G) 0
    hne : (Set.range orderOf).Nonempty
    hfin : (Set.range orderOf).Finite
    t : G
    ht : Eq (orderOf t) (SupSet.sSup (Set.range orderOf))
    p : Nat
    hp : Nat.Prime p
    k : Nat := (orderOf t).factorization p
    hpe : LT.lt k ((Monoid.exponent G).factorization p)
    hk : Eq k ((orderOf t).factorization p)
    g : G
    hg : Eq (orderOf g) (HPow.hPow p ((Monoid.exponent G).factorization p))
    hpk : Dvd.dvd (HPow.hPow p k) (orderOf t)
    hpk' : Eq (orderOf (HPow.hPow t (HPow.hPow p k))) (HDiv.hDiv (orderOf t) (HPow …
    ⊢ LT.lt (orderOf t) (orderOf (HMul.hMul (HPow.hPow t (HPow.hPow p k)) g))
  -/
  obtain ⟨a, ha⟩ := Nat.exists_eq_add_of_lt hpe
  have hcoprime : (orderOf (t ^ p ^ k)).Coprime (orderOf g) := by
    rw [hg, Nat.coprime_pow_right_iff (pos_of_gt hpe), Nat.coprime_comm]
    apply Or.resolve_right (Nat.coprime_or_dvd_of_prime hp _)
    nth_rw 1 [← pow_one p]
    have : 1 = (Nat.factorization (orderOf (t ^ p ^ k))) p + 1 := by
     rw [hpk', Nat.factorization_div hpk]
     simp [k, hp]
    rw [this]
    -- Porting note: convert made to_additive complain
    apply Nat.pow_succ_factorization_not_dvd (hG.orderOf_pos <| t ^ p ^ k).ne' hp
  rw [(Commute.all _ g).orderOf_mul_eq_mul_orderOf_of_coprime hcoprime, hpk',
    hg, ha, hk, pow_add, pow_add, pow_one, ← mul_assoc, ← mul_assoc,
    Nat.div_mul_cancel, mul_assoc, lt_mul_iff_one_lt_right <| hG.orderOf_pos t, ← pow_succ]
    /-
      case h.intro.intro.intro.intro
      G : Type u
      inst✝ : CommMonoid G
      hG : Monoid.ExponentExists G
      he : Ne (Monoid.exponent G) 0
      hne : (Set.range orderOf).Nonempty
      hfin : (Set.range orderOf).Finite
      t : G
      ht : Eq (orderOf t) (SupSet.sSup (Set.range orderOf))
      p : Nat
      hp : Nat.Prime p
      k : Nat := (orderOf t).factorization p
      hpe : LT.lt k ((Monoid.exponent G).factorization p)
      hk : Eq k ((orderOf t).factorization p)
      g : G
      hg : Eq (orderOf g) (HPow.hPow p ((Monoid.exponent G).factorization p))
      hpk : Dvd.dvd (HPow.hPow p k) (orderOf t)
      hpk' : Eq (orderOf (HPow.hPow t (HPow.hPow p k))) (HDiv.hDiv (orderOf t) (HPow …
      a : Nat
      ha : Eq ((Monoid.exponent G).factorization p) (HAdd.hAdd (HAdd.hAdd k a) 1)
      hcoprime : (orderOf (HPow.hPow t (HPow.hPow p k))).Coprime (orderOf g)
      ⊢ LT.lt 1 (HPow.hPow p (HAdd.hAdd a 1))
    -/
  · exact one_lt_pow₀ hp.one_lt a.succ_ne_zero
    /-
      🎉 no goals
    -/
    /-
      case h.intro.intro.intro.intro
      G : Type u
      inst✝ : CommMonoid G
      hG : Monoid.ExponentExists G
      he : Ne (Monoid.exponent G) 0
      hne : (Set.range orderOf).Nonempty
      hfin : (Set.range orderOf).Finite
      t : G
      ht : Eq (orderOf t) (SupSet.sSup (Set.range orderOf))
      p : Nat
      hp : Nat.Prime p
      k : Nat := (orderOf t).factorization p
      hpe : LT.lt k ((Monoid.exponent G).factorization p)
      hk : Eq k ((orderOf t).factorization p)
      g : G
      hg : Eq (orderOf g) (HPow.hPow p ((Monoid.exponent G).factorization p))
      hpk : Dvd.dvd (HPow.hPow p k) (orderOf t)
      hpk' : Eq (orderOf (HPow.hPow t (HPow.hPow p k))) (HDiv.hDiv (orderOf t) (HPow …
      a : Nat
      ha : Eq ((Monoid.exponent G).factorization p) (HAdd.hAdd (HAdd.hAdd k a) 1)
      hcoprime : (orderOf (HPow.hPow t (HPow.hPow p k))).Coprime (orderOf g)
      ⊢ Dvd.dvd (HPow.hPow p ((orderOf t).factorization p)) (orderOf t)
    -/
  · exact hpk
    /-
      🎉 no goals
    -/


@[to_additive]
theorem exponent_eq_iSup_orderOf (h : ∀ g : G, 0 < orderOf g) :
    exponent G = ⨆ g : G, orderOf g := by
  /-
    G : Type u
    inst✝ : CommMonoid G
    h : ∀ (g : G), LT.lt 0 (orderOf g)
    ⊢ Eq (Monoid.exponent G) (iSup fun g => orderOf g)
  -/
  rw [iSup]
  /-
    G : Type u
    inst✝ : CommMonoid G
    h : ∀ (g : G), LT.lt 0 (orderOf g)
    ⊢ Eq (Monoid.exponent G) (SupSet.sSup (Set.range fun g => orderOf g))
  -/
  by_cases ExponentExists G
  case neg he =>
    rw [← exponent_eq_zero_iff] at he
    rw [he, Set.Infinite.Nat.sSup_eq_zero <| (exponent_eq_zero_iff_range_orderOf_infinite h).1 he]
  case pos he =>
    rw [csSup_eq_of_forall_le_of_forall_lt_exists_gt (Set.range_nonempty _)]
    · simp_rw [Set.mem_range, forall_exists_index, forall_apply_eq_imp_iff]
      exact orderOf_le_exponent he
    intro x hx
    obtain ⟨g, hg⟩ := exists_orderOf_eq_exponent he
    rw [← hg] at hx
    simp_rw [Set.mem_range, exists_exists_eq_and]
    exact ⟨g, hx⟩


@[to_additive]
theorem exponent_eq_iSup_orderOf' :
    exponent G = if ∃ g : G, orderOf g = 0 then 0 else ⨆ g : G, orderOf g := by
  /-
    G : Type u
    inst✝ : CommMonoid G
    ⊢ Eq (Monoid.exponent G) (ite (Exists fun g => Eq (orderOf g) 0) 0 (iSup fun g …
  -/
  split_ifs with h
    /-
      case pos
      G : Type u
      inst✝ : CommMonoid G
      h : Exists fun g => Eq (orderOf g) 0
      ⊢ Eq (Monoid.exponent G) 0
    -/
  · obtain ⟨g, hg⟩ := h
    /-
      case pos.intro
      G : Type u
      inst✝ : CommMonoid G
      g : G
      hg : Eq (orderOf g) 0
      ⊢ Eq (Monoid.exponent G) 0
    -/
    exact exponent_eq_zero_of_order_zero hg
    /-
      🎉 no goals
    -/
    /-
      case neg
      G : Type u
      inst✝ : CommMonoid G
      h : Not (Exists fun g => Eq (orderOf g) 0)
      ⊢ Eq (Monoid.exponent G) (iSup fun g => orderOf g)
    -/
  · have := not_exists.mp h
    /-
      case neg
      G : Type u
      inst✝ : CommMonoid G
      h : Not (Exists fun g => Eq (orderOf g) 0)
      this : ∀ (x : G), Not (Eq (orderOf x) 0)
      ⊢ Eq (Monoid.exponent G) (iSup fun g => orderOf g)
    -/
    exact exponent_eq_iSup_orderOf fun g => Ne.bot_lt <| this g
    /-
      🎉 no goals
    -/


@[to_additive]
theorem exponent_eq_max'_orderOf [Fintype G] :
                                                                /-
                                                                  G : Type u
                                                                  inst✝¹ : CancelCommMonoid G
                                                                  inst✝ : Fintype G
                                                                  ⊢ Membership.mem (Finset.image orderOf Finset.univ) 1
                                                                -/
    exponent G = ((@Finset.univ G _).image orderOf).max' ⟨1, by simp⟩ := by
                                                                /-
                                                                  🎉 no goals
                                                                -/
  /-
    G : Type u
    inst✝¹ : CancelCommMonoid G
    inst✝ : Fintype G
    ⊢ Eq (Monoid.exponent G) ((Finset.image orderOf Finset.univ).max' ⋯)
  -/
  rw [← Finset.Nonempty.csSup_eq_max', Finset.coe_image, Finset.coe_univ, Set.image_univ, ← iSup]
  /-
    G : Type u
    inst✝¹ : CancelCommMonoid G
    inst✝ : Fintype G
    ⊢ Eq (Monoid.exponent G) (iSup orderOf)
  -/
  exact exponent_eq_iSup_orderOf orderOf_pos
  /-
    🎉 no goals
  -/


@[to_additive (attr := deprecated Monoid.one_lt_exponent (since := "2024-02-17"))
  AddGroup.one_lt_exponent]
lemma Group.one_lt_exponent [Finite G] [Nontrivial G] : 1 < Monoid.exponent G :=
  Monoid.one_lt_exponent


@[to_additive]
theorem Group.exponent_dvd_card [Fintype G] : Monoid.exponent G ∣ Fintype.card G :=
  Monoid.exponent_dvd.mpr <| fun _ => orderOf_dvd_card


@[to_additive]
theorem Group.exponent_dvd_nat_card : Monoid.exponent G ∣ Nat.card G :=
  Monoid.exponent_dvd.mpr orderOf_dvd_natCard


@[to_additive]
theorem Subgroup.exponent_toSubmonoid (H : Subgroup G) :
    Monoid.exponent H.toSubmonoid = Monoid.exponent H :=
  Monoid.exponent_eq_of_mulEquiv (MulEquiv.subgroupCongr rfl)


@[to_additive (attr := simp)]
theorem Subgroup.exponent_top : Monoid.exponent (⊤ : Subgroup G) = Monoid.exponent G :=
  Monoid.exponent_eq_of_mulEquiv topEquiv


@[to_additive]
theorem Subgroup.pow_exponent_eq_one {H : Subgroup G} {g : G} (g_in_H : g ∈ H) :
    g ^ Monoid.exponent H = 1 := exponent_toSubmonoid H ▸ Submonoid.pow_exponent_eq_one g_in_H


@[to_additive]
theorem Group.exponent_dvd_iff_forall_zpow_eq_one :
    (Monoid.exponent G : ℤ) ∣ n ↔ ∀ g : G, g ^ n = 1 := by
  /-
    G : Type u
    inst✝ : Group G
    n : Int
    ⊢ Iff (Dvd.dvd (↑(Monoid.exponent G)) n) (∀ (g : G), Eq (HPow.hPow g n) 1)
  -/
  simp_rw [Int.natCast_dvd, Monoid.exponent_dvd_iff_forall_pow_eq_one, pow_natAbs_eq_one]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Group.exponent_dvd_sub_iff_zpow_eq_zpow :
    (Monoid.exponent G : ℤ) ∣ n - m ↔ ∀ g : G, g ^ n = g ^ m := by
  /-
    G : Type u
    inst✝ : Group G
    n m : Int
    ⊢ Iff (Dvd.dvd (↑(Monoid.exponent G)) (HSub.hSub n m)) (∀ (g : G), Eq (HPow.hP …
  -/
  simp_rw [Group.exponent_dvd_iff_forall_zpow_eq_one, zpow_sub, mul_inv_eq_one]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Monoid.exponent_pi_eq_zero {ι : Type*} {M : ι → Type*} [∀ i, Monoid (M i)] {j : ι}
    (hj : exponent (M j) = 0) : exponent ((i : ι) → M i) = 0 := by
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    j : ι
    hj : Eq (Monoid.exponent (M j)) 0
    ⊢ Eq (Monoid.exponent ((i : ι) → M i)) 0
  -/
  rw [@exponent_eq_zero_iff, ExponentExists] at hj ⊢
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    j : ι
    hj : Not (Exists fun n => And (LT.lt 0 n) (∀ (g : M j), Eq (HPow.hPow g n) 1))
    ⊢ Not (Exists fun n => And (LT.lt 0 n) (∀ (g : (i : ι) → M i), Eq (HPow.hPow g …
  -/
  push_neg at hj ⊢
  /-
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    j : ι
    hj : ∀ (n : Nat), LT.lt 0 n → Exists fun g => Ne (HPow.hPow g n) 1
    ⊢ ∀ (n : Nat), LT.lt 0 n → Exists fun g => Ne (HPow.hPow g n) 1
  -/
  peel hj with n hn _
  /-
    case h.h
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    j : ι
    hj : ∀ (n : Nat), LT.lt 0 n → Exists fun g => Ne (HPow.hPow g n) 1
    n : Nat
    hn : LT.lt 0 n
    this : Exists fun g => Ne (HPow.hPow g n) 1
    ⊢ Exists fun g => Ne (HPow.hPow g n) 1
  -/
  obtain ⟨m, hm⟩ := this
  /-
    case h.h.intro
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    j : ι
    hj : ∀ (n : Nat), LT.lt 0 n → Exists fun g => Ne (HPow.hPow g n) 1
    n : Nat
    hn : LT.lt 0 n
    m : M j
    hm : Ne (HPow.hPow m n) 1
    ⊢ Exists fun g => Ne (HPow.hPow g n) 1
  -/
  refine ⟨Pi.mulSingle j m, fun h ↦ hm ?_⟩
  /-
    case h.h.intro
    ι : Type u_1
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    j : ι
    hj : ∀ (n : Nat), LT.lt 0 n → Exists fun g => Ne (HPow.hPow g n) 1
    n : Nat
    hn : LT.lt 0 n
    m : M j
    hm : Ne (HPow.hPow m n) 1
    h : Eq (HPow.hPow (Pi.mulSingle j m) n) 1
    ⊢ Eq (HPow.hPow m n) 1
  -/
  simpa using congr_fun h j
  /-
    🎉 no goals
  -/


/-- If `f : M₁ →⋆ M₂` is surjective, then the exponent of `M₂` divides the exponent of `M₁`. -/
@[to_additive]
theorem MonoidHom.exponent_dvd {F M₁ M₂ : Type*} [Monoid M₁] [Monoid M₂]
    [FunLike F M₁ M₂] [MonoidHomClass F M₁ M₂]
    {f : F} (hf : Function.Surjective f) : exponent M₂ ∣ exponent M₁ := by
  /-
    F : Type u_1
    M₁ : Type u_2
    M₂ : Type u_3
    inst✝³ : Monoid M₁
    inst✝² : Monoid M₂
    inst✝¹ : FunLike F M₁ M₂
    inst✝ : MonoidHomClass F M₁ M₂
    f : F
    hf : Function.Surjective ⇑f
    ⊢ Dvd.dvd (Monoid.exponent M₂) (Monoid.exponent M₁)
  -/
  refine Monoid.exponent_dvd_of_forall_pow_eq_one fun m₂ ↦ ?_
  /-
    F : Type u_1
    M₁ : Type u_2
    M₂ : Type u_3
    inst✝³ : Monoid M₁
    inst✝² : Monoid M₂
    inst✝¹ : FunLike F M₁ M₂
    inst✝ : MonoidHomClass F M₁ M₂
    f : F
    hf : Function.Surjective ⇑f
    m₂ : M₂
    ⊢ Eq (HPow.hPow m₂ (Monoid.exponent M₁)) 1
  -/
  obtain ⟨m₁, rfl⟩ := hf m₂
  /-
    case intro
    F : Type u_1
    M₁ : Type u_2
    M₂ : Type u_3
    inst✝³ : Monoid M₁
    inst✝² : Monoid M₂
    inst✝¹ : FunLike F M₁ M₂
    inst✝ : MonoidHomClass F M₁ M₂
    f : F
    hf : Function.Surjective ⇑f
    m₁ : M₁
    ⊢ Eq (HPow.hPow (f m₁) (Monoid.exponent M₁)) 1
  -/
  rw [← map_pow, pow_exponent_eq_one, map_one]
  /-
    🎉 no goals
  -/


/-- The exponent of finite product of monoids is the `Finset.lcm` of the exponents of the
constituent monoids. -/
@[to_additive "The exponent of finite product of additive monoids is the `Finset.lcm` of the
exponents of the constituent additive monoids."]
theorem Monoid.exponent_pi {ι : Type*} [Fintype ι] {M : ι → Type*} [∀ i, Monoid (M i)] :
    exponent ((i : ι) → M i) = lcm univ (exponent <| M ·) := by
  /-
    ι : Type u_1
    inst✝¹ : Fintype ι
    M : ι → Type u_2
    inst✝ : (i : ι) → Monoid (M i)
    ⊢ Eq (Monoid.exponent ((i : ι) → M i)) (Finset.univ.lcm fun x => Monoid.expone …
  -/
  refine dvd_antisymm ?_ ?_
    /-
      case refine_1
      ι : Type u_1
      inst✝¹ : Fintype ι
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      ⊢ Dvd.dvd (Monoid.exponent ((i : ι) → M i)) (Finset.univ.lcm fun x => Monoid.e …
    -/
  · refine exponent_dvd_of_forall_pow_eq_one fun m ↦ ?_
    /-
      case refine_1
      ι : Type u_1
      inst✝¹ : Fintype ι
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      m : (i : ι) → M i
      ⊢ Eq (HPow.hPow m (Finset.univ.lcm fun x => Monoid.exponent (M x))) 1
    -/
    ext i
    /-
      case refine_1.h
      ι : Type u_1
      inst✝¹ : Fintype ι
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      m : (i : ι) → M i
      i : ι
      ⊢ Eq (HPow.hPow m (Finset.univ.lcm fun x => Monoid.exponent (M x)) i) (1 i)
    -/
    rw [Pi.pow_apply, Pi.one_apply, ← orderOf_dvd_iff_pow_eq_one]
    /-
      case refine_1.h
      ι : Type u_1
      inst✝¹ : Fintype ι
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      m : (i : ι) → M i
      i : ι
      ⊢ Dvd.dvd (orderOf (m i)) (Finset.univ.lcm fun x => Monoid.exponent (M x))
    -/
    apply dvd_trans (Monoid.order_dvd_exponent (m i))
    /-
      case refine_1.h
      ι : Type u_1
      inst✝¹ : Fintype ι
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      m : (i : ι) → M i
      i : ι
      ⊢ Dvd.dvd (Monoid.exponent (M i)) (Finset.univ.lcm fun x => Monoid.exponent (M …
    -/
    exact Finset.dvd_lcm (mem_univ i)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      inst✝¹ : Fintype ι
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      ⊢ Dvd.dvd (Finset.univ.lcm fun x => Monoid.exponent (M x)) (Monoid.exponent (( …
    -/
  · apply Finset.lcm_dvd fun i _ ↦ ?_
    /-
      ι : Type u_1
      inst✝¹ : Fintype ι
      M : ι → Type u_2
      inst✝ : (i : ι) → Monoid (M i)
      i : ι
      x✝ : Membership.mem Finset.univ i
      ⊢ Dvd.dvd (Monoid.exponent (M i)) (Monoid.exponent ((i : ι) → M i))
    -/
    exact MonoidHom.exponent_dvd (f := Pi.evalMonoidHom (M ·) i) (Function.surjective_eval i)
    /-
      🎉 no goals
    -/


/-- The exponent of product of two monoids is the `lcm` of the exponents of the
individuaul monoids. -/
@[to_additive AddMonoid.exponent_prod "The exponent of product of two additive monoids is the `lcm`
of the exponents of the individuaul additive monoids."]
theorem Monoid.exponent_prod {M₁ M₂ : Type*} [Monoid M₁] [Monoid M₂] :
    exponent (M₁ × M₂) = lcm (exponent M₁) (exponent M₂) := by
  /-
    M₁ : Type u_1
    M₂ : Type u_2
    inst✝¹ : Monoid M₁
    inst✝ : Monoid M₂
    ⊢ Eq (Monoid.exponent (Prod M₁ M₂)) (GCDMonoid.lcm (Monoid.exponent M₁) (Monoi …
  -/
  refine dvd_antisymm ?_ (lcm_dvd ?_ ?_)
    /-
      case refine_1
      M₁ : Type u_1
      M₂ : Type u_2
      inst✝¹ : Monoid M₁
      inst✝ : Monoid M₂
      ⊢ Dvd.dvd (Monoid.exponent (Prod M₁ M₂)) (GCDMonoid.lcm (Monoid.exponent M₁) ( …
    -/
  · refine exponent_dvd_of_forall_pow_eq_one fun g ↦ ?_
    /-
      case refine_1
      M₁ : Type u_1
      M₂ : Type u_2
      inst✝¹ : Monoid M₁
      inst✝ : Monoid M₂
      g : Prod M₁ M₂
      ⊢ Eq (HPow.hPow g (GCDMonoid.lcm (Monoid.exponent M₁) (Monoid.exponent M₂))) 1
    -/
    ext1
      /-
        case refine_1.fst
        M₁ : Type u_1
        M₂ : Type u_2
        inst✝¹ : Monoid M₁
        inst✝ : Monoid M₂
        g : Prod M₁ M₂
        ⊢ Eq (HPow.hPow g (GCDMonoid.lcm (Monoid.exponent M₁) (Monoid.exponent M₂))).1 …
      -/
    · rw [Prod.pow_fst, Prod.fst_one, ← orderOf_dvd_iff_pow_eq_one]
      /-
        case refine_1.fst
        M₁ : Type u_1
        M₂ : Type u_2
        inst✝¹ : Monoid M₁
        inst✝ : Monoid M₂
        g : Prod M₁ M₂
        ⊢ Dvd.dvd (orderOf g.1) (GCDMonoid.lcm (Monoid.exponent M₁) (Monoid.exponent M …
      -/
      exact dvd_trans (Monoid.order_dvd_exponent (g.1)) <| dvd_lcm_left _ _
      /-
        🎉 no goals
      -/
      /-
        case refine_1.snd
        M₁ : Type u_1
        M₂ : Type u_2
        inst✝¹ : Monoid M₁
        inst✝ : Monoid M₂
        g : Prod M₁ M₂
        ⊢ Eq (HPow.hPow g (GCDMonoid.lcm (Monoid.exponent M₁) (Monoid.exponent M₂))).2 …
      -/
    · rw [Prod.pow_snd, Prod.snd_one, ← orderOf_dvd_iff_pow_eq_one]
      /-
        case refine_1.snd
        M₁ : Type u_1
        M₂ : Type u_2
        inst✝¹ : Monoid M₁
        inst✝ : Monoid M₂
        g : Prod M₁ M₂
        ⊢ Dvd.dvd (orderOf g.2) (GCDMonoid.lcm (Monoid.exponent M₁) (Monoid.exponent M …
      -/
      exact dvd_trans (Monoid.order_dvd_exponent (g.2)) <| dvd_lcm_right _ _
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      M₁ : Type u_1
      M₂ : Type u_2
      inst✝¹ : Monoid M₁
      inst✝ : Monoid M₂
      ⊢ Dvd.dvd (Monoid.exponent M₁) (Monoid.exponent (Prod M₁ M₂))
    -/
  · exact MonoidHom.exponent_dvd (f := MonoidHom.fst M₁ M₂) Prod.fst_surjective
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      M₁ : Type u_1
      M₂ : Type u_2
      inst✝¹ : Monoid M₁
      inst✝ : Monoid M₂
      ⊢ Dvd.dvd (Monoid.exponent M₂) (Monoid.exponent (Prod M₁ M₂))
    -/
  · exact MonoidHom.exponent_dvd (f := MonoidHom.snd M₁ M₂) Prod.snd_surjective
    /-
      🎉 no goals
    -/


@[to_additive]
lemma orderOf_eq_two_iff (hG : Monoid.exponent G = 2) {x : G} :
    orderOf x = 2 ↔ x ≠ 1 :=
      /-
        G : Type u
        inst✝ : Monoid G
        hG : Eq (Monoid.exponent G) 2
        x : G
        ⊢ Eq (orderOf x) 2 → Ne x 1
      -/
  ⟨by rintro hx rfl; norm_num at hx, orderOf_eq_prime (hG ▸ Monoid.pow_exponent_eq_one x)⟩
                     /-
                       🎉 no goals
                     -/


@[to_additive]
theorem Commute.of_orderOf_dvd_two [IsCancelMul G] (h : ∀ g : G, orderOf g ∣ 2) (a b : G) :
    Commute a b := by
  /-
    G : Type u
    inst✝¹ : Monoid G
    inst✝ : IsCancelMul G
    h : ∀ (g : G), Dvd.dvd (orderOf g) 2
    a b : G
    ⊢ Commute a b
  -/
  simp_rw [orderOf_dvd_iff_pow_eq_one] at h
  /-
    G : Type u
    inst✝¹ : Monoid G
    inst✝ : IsCancelMul G
    a b : G
    h : ∀ (g : G), Eq (HPow.hPow g 2) 1
    ⊢ Commute a b
  -/
  rw [commute_iff_eq, ← mul_right_inj a, ← mul_left_inj b]
  calc
    a * (a * b) * b = a ^ 2 * b ^ 2 := by simp only [pow_two]; group
    _ = 1 := by rw [h, h, mul_one]
    _ = (a * b) ^ 2 := by rw [h]
    _ = a * (b * a) * b := by simp only [pow_two]; group


/-- In a cancellative monoid of exponent two, all elements commute. -/
@[to_additive]
lemma mul_comm_of_exponent_two [IsCancelMul G] (hG : Monoid.exponent G = 2) (a b : G) :
    a * b = b * a :=
  Commute.of_orderOf_dvd_two (fun g => hG ▸ Monoid.order_dvd_exponent g) a b


/-- Any cancellative monoid of exponent two is abelian. -/
@[to_additive "Any additive group of exponent two is abelian."]
abbrev commMonoidOfExponentTwo [IsCancelMul G] (hG : Monoid.exponent G = 2) : CommMonoid G where
  mul_comm := mul_comm_of_exponent_two hG


/-- In a group of exponent two, every element is its own inverse. -/
@[to_additive]
lemma inv_eq_self_of_exponent_two (hG : Monoid.exponent G = 2) (x : G) :
    x⁻¹ = x :=
  inv_eq_of_mul_eq_one_left <| pow_two (a := x) ▸ hG ▸ Monoid.pow_exponent_eq_one x


/-- If an element in a group has order two, then it is its own inverse. -/
@[to_additive]
lemma inv_eq_self_of_orderOf_eq_two {x : G} (hx : orderOf x = 2) :
    x⁻¹ = x :=
  inv_eq_of_mul_eq_one_left <| pow_two (a := x) ▸ hx ▸ pow_orderOf_eq_one x

-- TODO: delete

/-- Any group of exponent two is abelian. -/
@[to_additive (attr := reducible,
  deprecated "No deprecation message was provided." (since := "2024-02-17"))
  "Any additive group of exponent two is abelian."]
def instCommGroupOfExponentTwo (hG : Monoid.exponent G = 2) : CommGroup G where
  mul_comm := mul_comm_of_exponent_two hG


@[to_additive]
lemma mul_not_mem_of_orderOf_eq_two {x y : G} (hx : orderOf x = 2)
    (hy : orderOf y = 2) (hxy : x ≠ y) : x * y ∉ ({x, y, 1} : Set G) := by
  simp only [Set.mem_singleton_iff, Set.mem_insert_iff, mul_right_eq_self, mul_left_eq_self,
    mul_eq_one_iff_eq_inv, inv_eq_self_of_orderOf_eq_two hy, not_or]
  /-
    G : Type u
    inst✝ : Group G
    x y : G
    hx : Eq (orderOf x) 2
    hy : Eq (orderOf y) 2
    hxy : Ne x y
    ⊢ And (Not (Eq y 1)) (And (Not (Eq x 1)) (Not (Eq x y)))
  -/
  aesop
  /-
    🎉 no goals
  -/


@[to_additive]
lemma mul_not_mem_of_exponent_two (h : Monoid.exponent G = 2) {x y : G}
    (hx : x ≠ 1) (hy : y ≠ 1) (hxy : x ≠ y) : x * y ∉ ({x, y, 1} : Set G) :=
  mul_not_mem_of_orderOf_eq_two (orderOf_eq_prime (h ▸ Monoid.pow_exponent_eq_one x) hx)
    (orderOf_eq_prime (h ▸ Monoid.pow_exponent_eq_one y) hy) hxy



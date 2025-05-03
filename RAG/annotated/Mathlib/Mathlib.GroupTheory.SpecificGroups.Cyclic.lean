@[to_additive]
theorem IsCyclic.exists_generator [Group α] [IsCyclic α] : ∃ g : α, ∀ x, x ∈ zpowers g :=
  exists_zpow_surjective α


@[to_additive]
theorem isCyclic_iff_exists_zpowers_eq_top [Group α] : IsCyclic α ↔ ∃ g : α, zpowers g = ⊤ := by
  /-
    α : Type u_1
    inst✝ : Group α
    ⊢ Iff (IsCyclic α) (Exists fun g => Eq (Subgroup.zpowers g) Top.top)
  -/
  simp only [eq_top_iff', mem_zpowers_iff]
  /-
    α : Type u_1
    inst✝ : Group α
    ⊢ Iff (IsCyclic α) (Exists fun g => ∀ (x : α), Exists fun k => Eq (HPow.hPow g …
  -/
  exact ⟨fun ⟨h⟩ ↦ h, fun h ↦ ⟨h⟩⟩
  /-
    🎉 no goals
  -/


@[to_additive]
instance (priority := 100) isCyclic_of_subsingleton [Group α] [Subsingleton α] : IsCyclic α :=
  ⟨⟨1, fun _ => ⟨0, Subsingleton.elim _ _⟩⟩⟩


@[simp]
theorem isCyclic_multiplicative_iff [AddGroup α] : IsCyclic (Multiplicative α) ↔ IsAddCyclic α :=
  ⟨fun H ↦ ⟨H.1⟩, fun H ↦ ⟨H.1⟩⟩


instance isCyclic_multiplicative [AddGroup α] [IsAddCyclic α] : IsCyclic (Multiplicative α) :=
  isCyclic_multiplicative_iff.mpr inferInstance


@[simp]
theorem isAddCyclic_additive_iff [Group α] : IsAddCyclic (Additive α) ↔ IsCyclic α :=
  ⟨fun H ↦ ⟨H.1⟩, fun H ↦ ⟨H.1⟩⟩


instance isAddCyclic_additive [Group α] [IsCyclic α] : IsAddCyclic (Additive α) :=
  isAddCyclic_additive_iff.mpr inferInstance


/-- A cyclic group is always commutative. This is not an `instance` because often we have a better
proof of `CommGroup`. -/
@[to_additive
      "A cyclic group is always commutative. This is not an `instance` because often we have
      a better proof of `AddCommGroup`."]
def IsCyclic.commGroup [hg : Group α] [IsCyclic α] : CommGroup α :=
  { hg with
    mul_comm := fun x y =>
      let ⟨_, hg⟩ := IsCyclic.exists_generator (α := α)
      let ⟨_, hn⟩ := hg x
      let ⟨_, hm⟩ := hg y
      hm ▸ hn ▸ zpow_mul_comm _ _ _ }


instance [Group G] (H : Subgroup G) [IsCyclic H] : H.IsCommutative :=
  ⟨⟨IsCyclic.commGroup.mul_comm⟩⟩


/-- A non-cyclic multiplicative group is non-trivial. -/
@[to_additive "A non-cyclic additive group is non-trivial."]
theorem Nontrivial.of_not_isCyclic (nc : ¬IsCyclic α) : Nontrivial α := by
  /-
    α : Type u_1
    inst✝ : Group α
    nc : Not (IsCyclic α)
    ⊢ Nontrivial α
  -/
  contrapose! nc
  /-
    α : Type u_1
    inst✝ : Group α
    nc : Not (Nontrivial α)
    ⊢ IsCyclic α
  -/
  exact @isCyclic_of_subsingleton _ _ (not_nontrivial_iff_subsingleton.mp nc)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem MonoidHom.map_cyclic [h : IsCyclic G] (σ : G →* G) :
    ∃ m : ℤ, ∀ g : G, σ g = g ^ m := by
  /-
    G : Type u_2
    inst✝ : Group G
    h : IsCyclic G
    σ : MonoidHom G G
    ⊢ Exists fun m => ∀ (g : G), Eq (σ g) (HPow.hPow g m)
  -/
  obtain ⟨h, hG⟩ := IsCyclic.exists_generator (α := G)
  /-
    case intro
    G : Type u_2
    inst✝ : Group G
    h✝ : IsCyclic G
    σ : MonoidHom G G
    h : G
    hG : ∀ (x : G), Membership.mem (Subgroup.zpowers h) x
    ⊢ Exists fun m => ∀ (g : G), Eq (σ g) (HPow.hPow g m)
  -/
  obtain ⟨m, hm⟩ := hG (σ h)
  /-
    case intro.intro
    G : Type u_2
    inst✝ : Group G
    h✝ : IsCyclic G
    σ : MonoidHom G G
    h : G
    hG : ∀ (x : G), Membership.mem (Subgroup.zpowers h) x
    m : Int
    hm : Eq ((fun x => HPow.hPow h x) m) (σ h)
    ⊢ Exists fun m => ∀ (g : G), Eq (σ g) (HPow.hPow g m)
  -/
  refine ⟨m, fun g => ?_⟩
  /-
    case intro.intro
    G : Type u_2
    inst✝ : Group G
    h✝ : IsCyclic G
    σ : MonoidHom G G
    h : G
    hG : ∀ (x : G), Membership.mem (Subgroup.zpowers h) x
    m : Int
    hm : Eq ((fun x => HPow.hPow h x) m) (σ h)
    g : G
    ⊢ Eq (σ g) (HPow.hPow g m)
  -/
  obtain ⟨n, rfl⟩ := hG g
  /-
    case intro.intro.intro
    G : Type u_2
    inst✝ : Group G
    h✝ : IsCyclic G
    σ : MonoidHom G G
    h : G
    hG : ∀ (x : G), Membership.mem (Subgroup.zpowers h) x
    m : Int
    hm : Eq ((fun x => HPow.hPow h x) m) (σ h)
    n : Int
    ⊢ Eq (σ ((fun x => HPow.hPow h x) n)) (HPow.hPow ((fun x => HPow.hPow h x) n) m)
  -/
  rw [MonoidHom.map_zpow, ← hm, ← zpow_mul, ← zpow_mul']
  /-
    🎉 no goals
  -/

@[deprecated (since := "2024-02-21")] alias
MonoidAddHom.map_add_cyclic := AddMonoidHom.map_addCyclic


@[to_additive]
lemma isCyclic_iff_exists_orderOf_eq_natCard [Finite α] :
    IsCyclic α ↔ ∃ g : α, orderOf g = Nat.card α := by
  /-
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : Finite α
    ⊢ Iff (IsCyclic α) (Exists fun g => Eq (orderOf g) (Nat.card α))
  -/
  simp_rw [isCyclic_iff_exists_zpowers_eq_top, ← card_eq_iff_eq_top, Nat.card_zpowers]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-20")]
alias isCyclic_iff_exists_ofOrder_eq_natCard := isCyclic_iff_exists_orderOf_eq_natCard


@[deprecated (since := "2024-12-20")]
alias isAddCyclic_iff_exists_ofOrder_eq_natCard := isAddCyclic_iff_exists_addOrderOf_eq_natCard


@[deprecated (since := "2024-12-20")]
alias IsCyclic.iff_exists_ofOrder_eq_natCard_of_Fintype :=
  isCyclic_iff_exists_orderOf_eq_natCard


@[deprecated (since := "2024-12-20")]
alias IsAddCyclic.iff_exists_ofOrder_eq_natCard_of_Fintype :=
  isAddCyclic_iff_exists_addOrderOf_eq_natCard


@[to_additive]
theorem isCyclic_of_orderOf_eq_card [Finite α] (x : α) (hx : orderOf x = Nat.card α) :
    IsCyclic α :=
  isCyclic_iff_exists_orderOf_eq_natCard.mpr ⟨x, hx⟩


@[deprecated (since := "2024-02-21")]
alias isAddCyclic_of_orderOf_eq_card := isAddCyclic_of_addOrderOf_eq_card


@[to_additive]
theorem Subgroup.eq_bot_or_eq_top_of_prime_card
    (H : Subgroup G) [hp : Fact (Nat.card G).Prime] : H = ⊥ ∨ H = ⊤ := by
  /-
    G : Type u_2
    inst✝ : Group G
    H : Subgroup G
    hp : Fact (Nat.Prime (Nat.card G))
    ⊢ Or (Eq H Bot.bot) (Eq H Top.top)
  -/
  have : Finite G := Nat.finite_of_card_ne_zero hp.1.ne_zero
  /-
    G : Type u_2
    inst✝ : Group G
    H : Subgroup G
    hp : Fact (Nat.Prime (Nat.card G))
    this : Finite G
    ⊢ Or (Eq H Bot.bot) (Eq H Top.top)
  -/
  have := card_subgroup_dvd_card H
  /-
    G : Type u_2
    inst✝ : Group G
    H : Subgroup G
    hp : Fact (Nat.Prime (Nat.card G))
    this✝ : Finite G
    this : Dvd.dvd (Nat.card (Subtype fun x => Membership.mem H x)) (Nat.card G)
    ⊢ Or (Eq H Bot.bot) (Eq H Top.top)
  -/
  rwa [Nat.dvd_prime hp.1, ← eq_bot_iff_card, card_eq_iff_eq_top] at this
  /-
    🎉 no goals
  -/


/-- Any non-identity element of a finite group of prime order generates the group. -/
@[to_additive "Any non-identity element of a finite group of prime order generates the group."]
theorem zpowers_eq_top_of_prime_card {p : ℕ}
    [hp : Fact p.Prime] (h : Nat.card G = p) {g : G} (hg : g ≠ 1) : zpowers g = ⊤ := by
  /-
    G : Type u_2
    inst✝ : Group G
    p : Nat
    hp : Fact (Nat.Prime p)
    h : Eq (Nat.card G) p
    g : G
    hg : Ne g 1
    ⊢ Eq (Subgroup.zpowers g) Top.top
  -/
  subst h
  /-
    G : Type u_2
    inst✝ : Group G
    g : G
    hg : Ne g 1
    hp : Fact (Nat.Prime (Nat.card G))
    ⊢ Eq (Subgroup.zpowers g) Top.top
  -/
  have := (zpowers g).eq_bot_or_eq_top_of_prime_card
  /-
    G : Type u_2
    inst✝ : Group G
    g : G
    hg : Ne g 1
    hp : Fact (Nat.Prime (Nat.card G))
    this : Or (Eq (Subgroup.zpowers g) Bot.bot) (Eq (Subgroup.zpowers g) Top.top)
    ⊢ Eq (Subgroup.zpowers g) Top.top
  -/
  rwa [zpowers_eq_bot, or_iff_right hg] at this
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mem_zpowers_of_prime_card {p : ℕ} [hp : Fact p.Prime]
    (h : Nat.card G = p) {g g' : G} (hg : g ≠ 1) : g' ∈ zpowers g := by
  /-
    G : Type u_2
    inst✝ : Group G
    p : Nat
    hp : Fact (Nat.Prime p)
    h : Eq (Nat.card G) p
    g g' : G
    hg : Ne g 1
    ⊢ Membership.mem (Subgroup.zpowers g) g'
  -/
  simp_rw [zpowers_eq_top_of_prime_card h hg, Subgroup.mem_top]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem mem_powers_of_prime_card {p : ℕ} [hp : Fact p.Prime]
    (h : Nat.card G = p) {g g' : G} (hg : g ≠ 1) : g' ∈ Submonoid.powers g := by
  /-
    G : Type u_2
    inst✝ : Group G
    p : Nat
    hp : Fact (Nat.Prime p)
    h : Eq (Nat.card G) p
    g g' : G
    hg : Ne g 1
    ⊢ Membership.mem (Submonoid.powers g) g'
  -/
  have : Finite G := Nat.finite_of_card_ne_zero (h ▸ hp.1.ne_zero)
  /-
    G : Type u_2
    inst✝ : Group G
    p : Nat
    hp : Fact (Nat.Prime p)
    h : Eq (Nat.card G) p
    g g' : G
    hg : Ne g 1
    this : Finite G
    ⊢ Membership.mem (Submonoid.powers g) g'
  -/
  rw [mem_powers_iff_mem_zpowers]
  /-
    G : Type u_2
    inst✝ : Group G
    p : Nat
    hp : Fact (Nat.Prime p)
    h : Eq (Nat.card G) p
    g g' : G
    hg : Ne g 1
    this : Finite G
    ⊢ Membership.mem (Subgroup.zpowers g) g'
  -/
  exact mem_zpowers_of_prime_card h hg
  /-
    🎉 no goals
  -/


@[to_additive]
theorem powers_eq_top_of_prime_card {p : ℕ}
    [hp : Fact p.Prime] (h : Nat.card G = p) {g : G} (hg : g ≠ 1) : Submonoid.powers g = ⊤ := by
  /-
    G : Type u_2
    inst✝ : Group G
    p : Nat
    hp : Fact (Nat.Prime p)
    h : Eq (Nat.card G) p
    g : G
    hg : Ne g 1
    ⊢ Eq (Submonoid.powers g) Top.top
  -/
  ext x
  /-
    case h
    G : Type u_2
    inst✝ : Group G
    p : Nat
    hp : Fact (Nat.Prime p)
    h : Eq (Nat.card G) p
    g : G
    hg : Ne g 1
    x : G
    ⊢ Iff (Membership.mem (Submonoid.powers g) x) (Membership.mem Top.top x)
  -/
  simp [mem_powers_of_prime_card h hg]
  /-
    🎉 no goals
  -/


/-- A finite group of prime order is cyclic. -/
@[to_additive "A finite group of prime order is cyclic."]
theorem isCyclic_of_prime_card {p : ℕ} [hp : Fact p.Prime]
    (h : Nat.card α = p) : IsCyclic α := by
  /-
    α : Type u_1
    inst✝ : Group α
    p : Nat
    hp : Fact (Nat.Prime p)
    h : Eq (Nat.card α) p
    ⊢ IsCyclic α
  -/
  have : Finite α := Nat.finite_of_card_ne_zero (h ▸ hp.1.ne_zero)
  /-
    α : Type u_1
    inst✝ : Group α
    p : Nat
    hp : Fact (Nat.Prime p)
    h : Eq (Nat.card α) p
    this : Finite α
    ⊢ IsCyclic α
  -/
  have : Nontrivial α := Finite.one_lt_card_iff_nontrivial.mp (h ▸ hp.1.one_lt)
  /-
    α : Type u_1
    inst✝ : Group α
    p : Nat
    hp : Fact (Nat.Prime p)
    h : Eq (Nat.card α) p
    this✝ : Finite α
    this : Nontrivial α
    ⊢ IsCyclic α
  -/
  obtain ⟨g, hg⟩ : ∃ g : α, g ≠ 1 := exists_ne 1
  /-
    case intro
    α : Type u_1
    inst✝ : Group α
    p : Nat
    hp : Fact (Nat.Prime p)
    h : Eq (Nat.card α) p
    this✝ : Finite α
    this : Nontrivial α
    g : α
    hg : Ne g 1
    ⊢ IsCyclic α
  -/
  exact ⟨g, fun g' ↦ mem_zpowers_of_prime_card h hg⟩
  /-
    🎉 no goals
  -/


/-- A finite group of order dividing a prime is cyclic. -/
@[to_additive "A finite group of order dividing a prime is cyclic."]
theorem isCyclic_of_card_dvd_prime {p : ℕ} [hp : Fact p.Prime]
    (h : Nat.card α ∣ p) : IsCyclic α := by
  /-
    α : Type u_1
    inst✝ : Group α
    p : Nat
    hp : Fact (Nat.Prime p)
    h : Dvd.dvd (Nat.card α) p
    ⊢ IsCyclic α
  -/
  rcases (Nat.dvd_prime hp.out).mp h with h | h
    /-
      case inl
      α : Type u_1
      inst✝ : Group α
      p : Nat
      hp : Fact (Nat.Prime p)
      h✝ : Dvd.dvd (Nat.card α) p
      h : Eq (Nat.card α) 1
      ⊢ IsCyclic α
    -/
  · exact @isCyclic_of_subsingleton α _ (Nat.card_eq_one_iff_unique.mp h).1
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : Group α
      p : Nat
      hp : Fact (Nat.Prime p)
      h✝ : Dvd.dvd (Nat.card α) p
      h : Eq (Nat.card α) p
      ⊢ IsCyclic α
    -/
  · exact isCyclic_of_prime_card h
    /-
      🎉 no goals
    -/


@[to_additive]
theorem isCyclic_of_surjective {F : Type*} [hH : IsCyclic G']
    [FunLike F G' G] [MonoidHomClass F G' G] (f : F) (hf : Function.Surjective f) :
    IsCyclic G := by
  /-
    G : Type u_2
    G' : Type u_3
    inst✝³ : Group G
    inst✝² : Group G'
    F : Type u_4
    hH : IsCyclic G'
    inst✝¹ : FunLike F G' G
    inst✝ : MonoidHomClass F G' G
    f : F
    hf : Function.Surjective ⇑f
    ⊢ IsCyclic G
  -/
  obtain ⟨x, hx⟩ := hH
  /-
    case mk.intro
    G : Type u_2
    G' : Type u_3
    inst✝³ : Group G
    inst✝² : Group G'
    F : Type u_4
    inst✝¹ : FunLike F G' G
    inst✝ : MonoidHomClass F G' G
    f : F
    hf : Function.Surjective ⇑f
    x : G'
    hx : Function.Surjective fun x_1 => HPow.hPow x x_1
    ⊢ IsCyclic G
  -/
  refine ⟨f x, fun a ↦ ?_⟩
  /-
    case mk.intro
    G : Type u_2
    G' : Type u_3
    inst✝³ : Group G
    inst✝² : Group G'
    F : Type u_4
    inst✝¹ : FunLike F G' G
    inst✝ : MonoidHomClass F G' G
    f : F
    hf : Function.Surjective ⇑f
    x : G'
    hx : Function.Surjective fun x_1 => HPow.hPow x x_1
    a : G
    ⊢ Exists fun a_1 => Eq ((fun x_1 => HPow.hPow (f x) x_1) a_1) a
  -/
  obtain ⟨a, rfl⟩ := hf a
  /-
    case mk.intro.intro
    G : Type u_2
    G' : Type u_3
    inst✝³ : Group G
    inst✝² : Group G'
    F : Type u_4
    inst✝¹ : FunLike F G' G
    inst✝ : MonoidHomClass F G' G
    f : F
    hf : Function.Surjective ⇑f
    x : G'
    hx : Function.Surjective fun x_1 => HPow.hPow x x_1
    a : G'
    ⊢ Exists fun a_1 => Eq ((fun x_1 => HPow.hPow (f x) x_1) a_1) (f a)
  -/
  obtain ⟨n, rfl⟩ := hx a
  /-
    case mk.intro.intro.intro
    G : Type u_2
    G' : Type u_3
    inst✝³ : Group G
    inst✝² : Group G'
    F : Type u_4
    inst✝¹ : FunLike F G' G
    inst✝ : MonoidHomClass F G' G
    f : F
    hf : Function.Surjective ⇑f
    x : G'
    hx : Function.Surjective fun x_1 => HPow.hPow x x_1
    n : Int
    ⊢ Exists fun a => Eq ((fun x_1 => HPow.hPow (f x) x_1) a) (f ((fun x_1 => HPow …
  -/
  exact ⟨n, (map_zpow _ _ _).symm⟩
  /-
    🎉 no goals
  -/


@[to_additive]
theorem orderOf_eq_card_of_forall_mem_zpowers {g : α} (hx : ∀ x, x ∈ zpowers g) :
    orderOf g = Nat.card α := by
  /-
    α : Type u_1
    inst✝ : Group α
    g : α
    hx : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
    ⊢ Eq (orderOf g) (Nat.card α)
  -/
  rw [← Nat.card_zpowers, (zpowers g).eq_top_iff'.mpr hx, card_top]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-15")]
alias orderOf_generator_eq_natCard := orderOf_eq_card_of_forall_mem_zpowers


@[deprecated (since := "2024-11-15")]
alias addOrderOf_generator_eq_natCard := addOrderOf_eq_card_of_forall_mem_zmultiples


@[to_additive]
theorem exists_pow_ne_one_of_isCyclic [G_cyclic : IsCyclic G]
    {k : ℕ} (k_pos : k ≠ 0) (k_lt_card_G : k < Nat.card G) : ∃ a : G, a ^ k ≠ 1 := by
  /-
    G : Type u_2
    inst✝ : Group G
    G_cyclic : IsCyclic G
    k : Nat
    k_pos : Ne k 0
    k_lt_card_G : LT.lt k (Nat.card G)
    ⊢ Exists fun a => Ne (HPow.hPow a k) 1
  -/
  have : Finite G := Nat.finite_of_card_ne_zero (Nat.not_eq_zero_of_lt k_lt_card_G)
  /-
    G : Type u_2
    inst✝ : Group G
    G_cyclic : IsCyclic G
    k : Nat
    k_pos : Ne k 0
    k_lt_card_G : LT.lt k (Nat.card G)
    this : Finite G
    ⊢ Exists fun a => Ne (HPow.hPow a k) 1
  -/
  rcases G_cyclic with ⟨a, ha⟩
  /-
    case mk.intro
    G : Type u_2
    inst✝ : Group G
    k : Nat
    k_pos : Ne k 0
    k_lt_card_G : LT.lt k (Nat.card G)
    this : Finite G
    a : G
    ha : Function.Surjective fun x => HPow.hPow a x
    ⊢ Exists fun a => Ne (HPow.hPow a k) 1
  -/
  use a
  /-
    case h
    G : Type u_2
    inst✝ : Group G
    k : Nat
    k_pos : Ne k 0
    k_lt_card_G : LT.lt k (Nat.card G)
    this : Finite G
    a : G
    ha : Function.Surjective fun x => HPow.hPow a x
    ⊢ Ne (HPow.hPow a k) 1
  -/
  contrapose! k_lt_card_G
  /-
    case h
    G : Type u_2
    inst✝ : Group G
    k : Nat
    k_pos : Ne k 0
    this : Finite G
    a : G
    ha : Function.Surjective fun x => HPow.hPow a x
    k_lt_card_G : Eq (HPow.hPow a k) 1
    ⊢ LE.le (Nat.card G) k
  -/
  convert orderOf_le_of_pow_eq_one k_pos.bot_lt k_lt_card_G
  /-
    case h.e'_3
    G : Type u_2
    inst✝ : Group G
    k : Nat
    k_pos : Ne k 0
    this : Finite G
    a : G
    ha : Function.Surjective fun x => HPow.hPow a x
    k_lt_card_G : Eq (HPow.hPow a k) 1
    ⊢ Eq (Nat.card G) (orderOf a)
  -/
  rw [← Nat.card_zpowers, eq_comm, card_eq_iff_eq_top, eq_top_iff]
  /-
    case h.e'_3
    G : Type u_2
    inst✝ : Group G
    k : Nat
    k_pos : Ne k 0
    this : Finite G
    a : G
    ha : Function.Surjective fun x => HPow.hPow a x
    k_lt_card_G : Eq (HPow.hPow a k) 1
    ⊢ LE.le Top.top (Subgroup.zpowers a)
  -/
  exact fun x _ ↦ ha x
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Infinite.orderOf_eq_zero_of_forall_mem_zpowers [Infinite α] {g : α}
    (h : ∀ x, x ∈ zpowers g) : orderOf g = 0 := by
  /-
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : Infinite α
    g : α
    h : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
    ⊢ Eq (orderOf g) 0
  -/
  rw [orderOf_eq_card_of_forall_mem_zpowers h, Nat.card_eq_zero_of_infinite]
  /-
    🎉 no goals
  -/


@[to_additive]
instance Bot.isCyclic : IsCyclic (⊥ : Subgroup α) :=
  ⟨⟨1, fun x => ⟨0, Subtype.eq <| (zpow_zero (1 : α)).trans <| Eq.symm (Subgroup.mem_bot.1 x.2)⟩⟩⟩


@[to_additive]
instance Subgroup.isCyclic [IsCyclic α] (H : Subgroup α) : IsCyclic H :=
  haveI := Classical.propDecidable
  let ⟨g, hg⟩ := IsCyclic.exists_generator (α := α)
  if hx : ∃ x : α, x ∈ H ∧ x ≠ (1 : α) then
    let ⟨x, hx₁, hx₂⟩ := hx
    let ⟨k, hk⟩ := hg x
    have hk : g ^ k = x := hk
    have hex : ∃ n : ℕ, 0 < n ∧ g ^ n ∈ H :=
      ⟨k.natAbs,
        Nat.pos_of_ne_zero fun h => hx₂ <| by
          /-
            α : Type u_1
            G : Type u_2
            G' : Type u_3
            a : α
            inst✝³ : Group α
            inst✝² : Group G
            inst✝¹ : Group G'
            inst✝ : IsCyclic α
            H : Subgroup α
            this : (a : Prop) → Decidable a
            g : α
            hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
            hx : Exists fun x => And (Membership.mem H x) (Ne x 1)
            x : α
            hx₁ : Membership.mem H x
            hx₂ : Ne x 1
            k : Int
            hk✝ : Eq ((fun x => HPow.hPow g x) k) x
            hk : Eq (HPow.hPow g k) x
            h : Eq k.natAbs 0
            ⊢ Eq x 1
          -/
          rw [← hk, Int.natAbs_eq_zero.mp h, zpow_zero], by
          /-
            🎉 no goals
          -/
            /-
              α : Type u_1
              G : Type u_2
              G' : Type u_3
              a : α
              inst✝³ : Group α
              inst✝² : Group G
              inst✝¹ : Group G'
              inst✝ : IsCyclic α
              H : Subgroup α
              this : (a : Prop) → Decidable a
              g : α
              hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
              hx : Exists fun x => And (Membership.mem H x) (Ne x 1)
              x : α
              hx₁ : Membership.mem H x
              hx₂ : Ne x 1
              k : Int
              hk✝ : Eq ((fun x => HPow.hPow g x) k) x
              hk : Eq (HPow.hPow g k) x
              ⊢ Membership.mem H (HPow.hPow g k.natAbs)
            -/
            cases' k with k k
              /-
                case ofNat
                α : Type u_1
                G : Type u_2
                G' : Type u_3
                a : α
                inst✝³ : Group α
                inst✝² : Group G
                inst✝¹ : Group G'
                inst✝ : IsCyclic α
                H : Subgroup α
                this : (a : Prop) → Decidable a
                g : α
                hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
                hx : Exists fun x => And (Membership.mem H x) (Ne x 1)
                x : α
                hx₁ : Membership.mem H x
                hx₂ : Ne x 1
                k : Nat
                hk✝ : Eq ((fun x => HPow.hPow g x) (Int.ofNat k)) x
                hk : Eq (HPow.hPow g (Int.ofNat k)) x
                ⊢ Membership.mem H (HPow.hPow g (Int.ofNat k).natAbs)
              -/
            · rw [Int.ofNat_eq_coe, Int.natAbs_cast k, ← zpow_natCast, ← Int.ofNat_eq_coe, hk]
              /-
                case ofNat
                α : Type u_1
                G : Type u_2
                G' : Type u_3
                a : α
                inst✝³ : Group α
                inst✝² : Group G
                inst✝¹ : Group G'
                inst✝ : IsCyclic α
                H : Subgroup α
                this : (a : Prop) → Decidable a
                g : α
                hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
                hx : Exists fun x => And (Membership.mem H x) (Ne x 1)
                x : α
                hx₁ : Membership.mem H x
                hx₂ : Ne x 1
                k : Nat
                hk✝ : Eq ((fun x => HPow.hPow g x) (Int.ofNat k)) x
                hk : Eq (HPow.hPow g (Int.ofNat k)) x
                ⊢ Membership.mem H x
              -/
              exact hx₁
              /-
                🎉 no goals
              -/
              /-
                case negSucc
                α : Type u_1
                G : Type u_2
                G' : Type u_3
                a : α
                inst✝³ : Group α
                inst✝² : Group G
                inst✝¹ : Group G'
                inst✝ : IsCyclic α
                H : Subgroup α
                this : (a : Prop) → Decidable a
                g : α
                hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
                hx : Exists fun x => And (Membership.mem H x) (Ne x 1)
                x : α
                hx₁ : Membership.mem H x
                hx₂ : Ne x 1
                k : Nat
                hk✝ : Eq ((fun x => HPow.hPow g x) (Int.negSucc k)) x
                hk : Eq (HPow.hPow g (Int.negSucc k)) x
                ⊢ Membership.mem H (HPow.hPow g (Int.negSucc k).natAbs)
              -/
            · rw [Int.natAbs_negSucc, ← Subgroup.inv_mem_iff H]; simp_all⟩
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    ⟨⟨⟨g ^ Nat.find hex, (Nat.find_spec hex).2⟩, fun ⟨x, hx⟩ =>
        let ⟨k, hk⟩ := hg x
        have hk : g ^ k = x := hk
        have hk₂ : g ^ ((Nat.find hex : ℤ) * (k / Nat.find hex : ℤ)) ∈ H := by
          /-
            α : Type u_1
            G : Type u_2
            G' : Type u_3
            a : α
            inst✝³ : Group α
            inst✝² : Group G
            inst✝¹ : Group G'
            inst✝ : IsCyclic α
            H : Subgroup α
            this : (a : Prop) → Decidable a
            g : α
            hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
            hx✝ : Exists fun x => And (Membership.mem H x) (Ne x 1)
            x✝¹ : α
            hx₁ : Membership.mem H x✝¹
            hx₂ : Ne x✝¹ 1
            k✝ : Int
            hk✝² : Eq ((fun x => HPow.hPow g x) k✝) x✝¹
            hk✝¹ : Eq (HPow.hPow g k✝) x✝¹
            hex : Exists fun n => And (LT.lt 0 n) (Membership.mem H (HPow.hPow g n))
            x✝ : Subtype fun x => Membership.mem H x
            x : α
            hx : Membership.mem H x
            k : Int
            hk✝ : Eq ((fun x => HPow.hPow g x) k) x
            hk : Eq (HPow.hPow g k) x
            ⊢ Membership.mem H (HPow.hPow g (HMul.hMul (↑(Nat.find hex)) (HDiv.hDiv k ↑(Na …
          -/
          rw [zpow_mul]
          /-
            α : Type u_1
            G : Type u_2
            G' : Type u_3
            a : α
            inst✝³ : Group α
            inst✝² : Group G
            inst✝¹ : Group G'
            inst✝ : IsCyclic α
            H : Subgroup α
            this : (a : Prop) → Decidable a
            g : α
            hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
            hx✝ : Exists fun x => And (Membership.mem H x) (Ne x 1)
            x✝¹ : α
            hx₁ : Membership.mem H x✝¹
            hx₂ : Ne x✝¹ 1
            k✝ : Int
            hk✝² : Eq ((fun x => HPow.hPow g x) k✝) x✝¹
            hk✝¹ : Eq (HPow.hPow g k✝) x✝¹
            hex : Exists fun n => And (LT.lt 0 n) (Membership.mem H (HPow.hPow g n))
            x✝ : Subtype fun x => Membership.mem H x
            x : α
            hx : Membership.mem H x
            k : Int
            hk✝ : Eq ((fun x => HPow.hPow g x) k) x
            hk : Eq (HPow.hPow g k) x
            ⊢ Membership.mem H (HPow.hPow (HPow.hPow g ↑(Nat.find hex)) (HDiv.hDiv k ↑(Nat …
          -/
          apply H.zpow_mem
          /-
            case hx
            α : Type u_1
            G : Type u_2
            G' : Type u_3
            a : α
            inst✝³ : Group α
            inst✝² : Group G
            inst✝¹ : Group G'
            inst✝ : IsCyclic α
            H : Subgroup α
            this : (a : Prop) → Decidable a
            g : α
            hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
            hx✝ : Exists fun x => And (Membership.mem H x) (Ne x 1)
            x✝¹ : α
            hx₁ : Membership.mem H x✝¹
            hx₂ : Ne x✝¹ 1
            k✝ : Int
            hk✝² : Eq ((fun x => HPow.hPow g x) k✝) x✝¹
            hk✝¹ : Eq (HPow.hPow g k✝) x✝¹
            hex : Exists fun n => And (LT.lt 0 n) (Membership.mem H (HPow.hPow g n))
            x✝ : Subtype fun x => Membership.mem H x
            x : α
            hx : Membership.mem H x
            k : Int
            hk✝ : Eq ((fun x => HPow.hPow g x) k) x
            hk : Eq (HPow.hPow g k) x
            ⊢ Membership.mem H (HPow.hPow g ↑(Nat.find hex))
          -/
          exact mod_cast (Nat.find_spec hex).2
          /-
            🎉 no goals
          -/
        have hk₃ : g ^ (k % Nat.find hex : ℤ) ∈ H :=
          (Subgroup.mul_mem_cancel_right H hk₂).1 <| by
            /-
              α : Type u_1
              G : Type u_2
              G' : Type u_3
              a : α
              inst✝³ : Group α
              inst✝² : Group G
              inst✝¹ : Group G'
              inst✝ : IsCyclic α
              H : Subgroup α
              this : (a : Prop) → Decidable a
              g : α
              hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
              hx✝ : Exists fun x => And (Membership.mem H x) (Ne x 1)
              x✝¹ : α
              hx₁ : Membership.mem H x✝¹
              hx₂ : Ne x✝¹ 1
              k✝ : Int
              hk✝² : Eq ((fun x => HPow.hPow g x) k✝) x✝¹
              hk✝¹ : Eq (HPow.hPow g k✝) x✝¹
              hex : Exists fun n => And (LT.lt 0 n) (Membership.mem H (HPow.hPow g n))
              x✝ : Subtype fun x => Membership.mem H x
              x : α
              hx : Membership.mem H x
              k : Int
              hk✝ : Eq ((fun x => HPow.hPow g x) k) x
              hk : Eq (HPow.hPow g k) x
              hk₂ : Membership.mem H (HPow.hPow g (HMul.hMul (↑(Nat.find hex)) (HDiv.hDiv k  …
              ⊢ Membership.mem H (HMul.hMul (HPow.hPow g (HMod.hMod k ↑(Nat.find hex))) (HPo …
            -/
            rw [← zpow_add, Int.emod_add_ediv, hk]; exact hx
                                                    /-
                                                      🎉 no goals
                                                    -/
        have hk₄ : k % Nat.find hex = (k % Nat.find hex).natAbs := by
          rw [Int.natAbs_of_nonneg
              (Int.emod_nonneg _ (Int.natCast_ne_zero_iff_pos.2 (Nat.find_spec hex).1))]
                                                           /-
                                                             α : Type u_1
                                                             G : Type u_2
                                                             G' : Type u_3
                                                             a : α
                                                             inst✝³ : Group α
                                                             inst✝² : Group G
                                                             inst✝¹ : Group G'
                                                             inst✝ : IsCyclic α
                                                             H : Subgroup α
                                                             this : (a : Prop) → Decidable a
                                                             g : α
                                                             hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
                                                             hx✝ : Exists fun x => And (Membership.mem H x) (Ne x 1)
                                                             x✝¹ : α
                                                             hx₁ : Membership.mem H x✝¹
                                                             hx₂ : Ne x✝¹ 1
                                                             k✝ : Int
                                                             hk✝² : Eq ((fun x => HPow.hPow g x) k✝) x✝¹
                                                             hk✝¹ : Eq (HPow.hPow g k✝) x✝¹
                                                             hex : Exists fun n => And (LT.lt 0 n) (Membership.mem H (HPow.hPow g n))
                                                             x✝ : Subtype fun x => Membership.mem H x
                                                             x : α
                                                             hx : Membership.mem H x
                                                             k : Int
                                                             hk✝ : Eq ((fun x => HPow.hPow g x) k) x
                                                             hk : Eq (HPow.hPow g k) x
                                                             hk₂ : Membership.mem H (HPow.hPow g (HMul.hMul (↑(Nat.find hex)) (HDiv.hDiv k  …
                                                             hk₃ : Membership.mem H (HPow.hPow g (HMod.hMod k ↑(Nat.find hex)))
                                                             hk₄ : Eq (HMod.hMod k ↑(Nat.find hex)) ↑(HMod.hMod k ↑(Nat.find hex)).natAbs
                                                             ⊢ Membership.mem H (HPow.hPow g (HMod.hMod k ↑(Nat.find hex)).natAbs)
                                                           -/
        have hk₅ : g ^ (k % Nat.find hex).natAbs ∈ H := by rwa [← zpow_natCast, ← hk₄]
                                                           /-
                                                             🎉 no goals
                                                           -/
        have hk₆ : (k % (Nat.find hex : ℤ)).natAbs = 0 :=
          by_contradiction fun h =>
            Nat.find_min hex
              (Int.ofNat_lt.1 <| by
                /-
                  α : Type u_1
                  G : Type u_2
                  G' : Type u_3
                  a : α
                  inst✝³ : Group α
                  inst✝² : Group G
                  inst✝¹ : Group G'
                  inst✝ : IsCyclic α
                  H : Subgroup α
                  this : (a : Prop) → Decidable a
                  g : α
                  hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
                  hx✝ : Exists fun x => And (Membership.mem H x) (Ne x 1)
                  x✝¹ : α
                  hx₁ : Membership.mem H x✝¹
                  hx₂ : Ne x✝¹ 1
                  k✝ : Int
                  hk✝² : Eq ((fun x => HPow.hPow g x) k✝) x✝¹
                  hk✝¹ : Eq (HPow.hPow g k✝) x✝¹
                  hex : Exists fun n => And (LT.lt 0 n) (Membership.mem H (HPow.hPow g n))
                  x✝ : Subtype fun x => Membership.mem H x
                  x : α
                  hx : Membership.mem H x
                  k : Int
                  hk✝ : Eq ((fun x => HPow.hPow g x) k) x
                  hk : Eq (HPow.hPow g k) x
                  hk₂ : Membership.mem H (HPow.hPow g (HMul.hMul (↑(Nat.find hex)) (HDiv.hDiv k  …
                  hk₃ : Membership.mem H (HPow.hPow g (HMod.hMod k ↑(Nat.find hex)))
                  hk₄ : Eq (HMod.hMod k ↑(Nat.find hex)) ↑(HMod.hMod k ↑(Nat.find hex)).natAbs
                  hk₅ : Membership.mem H (HPow.hPow g (HMod.hMod k ↑(Nat.find hex)).natAbs)
                  h : Not (Eq (HMod.hMod k ↑(Nat.find hex)).natAbs 0)
                  ⊢ LT.lt ↑(HMod.hMod k ↑(Nat.find hex)).natAbs ↑(Nat.find hex)
                -/
                rw [← hk₄]; exact Int.emod_lt_of_pos _ (Int.natCast_pos.2 (Nat.find_spec hex).1))
                            /-
                              🎉 no goals
                            -/
              ⟨Nat.pos_of_ne_zero h, hk₅⟩
        ⟨k / (Nat.find hex : ℤ),
          Subtype.ext_iff_val.2
            (by
              /-
                α : Type u_1
                G : Type u_2
                G' : Type u_3
                a : α
                inst✝³ : Group α
                inst✝² : Group G
                inst✝¹ : Group G'
                inst✝ : IsCyclic α
                H : Subgroup α
                this : (a : Prop) → Decidable a
                g : α
                hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
                hx✝ : Exists fun x => And (Membership.mem H x) (Ne x 1)
                x✝¹ : α
                hx₁ : Membership.mem H x✝¹
                hx₂ : Ne x✝¹ 1
                k✝ : Int
                hk✝² : Eq ((fun x => HPow.hPow g x) k✝) x✝¹
                hk✝¹ : Eq (HPow.hPow g k✝) x✝¹
                hex : Exists fun n => And (LT.lt 0 n) (Membership.mem H (HPow.hPow g n))
                x✝ : Subtype fun x => Membership.mem H x
                x : α
                hx : Membership.mem H x
                k : Int
                hk✝ : Eq ((fun x => HPow.hPow g x) k) x
                hk : Eq (HPow.hPow g k) x
                hk₂ : Membership.mem H (HPow.hPow g (HMul.hMul (↑(Nat.find hex)) (HDiv.hDiv k  …
                hk₃ : Membership.mem H (HPow.hPow g (HMod.hMod k ↑(Nat.find hex)))
                hk₄ : Eq (HMod.hMod k ↑(Nat.find hex)) ↑(HMod.hMod k ↑(Nat.find hex)).natAbs
                hk₅ : Membership.mem H (HPow.hPow g (HMod.hMod k ↑(Nat.find hex)).natAbs)
                hk₆ : Eq (HMod.hMod k ↑(Nat.find hex)).natAbs 0
                ⊢ Eq ↑((fun x => HPow.hPow ⟨HPow.hPow g (Nat.find hex), ⋯⟩ x) (HDiv.hDiv k ↑(N …
              -/
              suffices g ^ ((Nat.find hex : ℤ) * (k / Nat.find hex : ℤ)) = x by simpa [zpow_mul]
              rw [Int.mul_ediv_cancel'
                  (Int.dvd_of_emod_eq_zero (Int.natAbs_eq_zero.mp hk₆)),
                hk])⟩⟩⟩
  else by
    have : H = (⊥ : Subgroup α) :=
      Subgroup.ext fun x =>
        ⟨fun h => by simp at *; tauto, fun h => by rw [Subgroup.mem_bot.1 h]; exact H.one_mem⟩
    /-
      α : Type u_1
      G : Type u_2
      G' : Type u_3
      a : α
      inst✝³ : Group α
      inst✝² : Group G
      inst✝¹ : Group G'
      inst✝ : IsCyclic α
      H : Subgroup α
      this✝ : (a : Prop) → Decidable a
      g : α
      hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
      hx : Not (Exists fun x => And (Membership.mem H x) (Ne x 1))
      this : Eq H Bot.bot
      ⊢ IsCyclic (Subtype fun x => Membership.mem H x)
    -/
    subst this; infer_instance
                /-
                  🎉 no goals
                -/


@[to_additive]
theorem isCyclic_of_injective [IsCyclic G'] (f : G →* G') (hf : Function.Injective f) :
    IsCyclic G :=
  isCyclic_of_surjective (MonoidHom.ofInjective hf).symm (MonoidHom.ofInjective hf).symm.surjective


@[to_additive]
lemma Subgroup.isCyclic_of_le {H H' : Subgroup G} (h : H ≤ H') [IsCyclic H'] : IsCyclic H :=
  isCyclic_of_injective (Subgroup.inclusion h) (Subgroup.inclusion_injective h)


@[to_additive IsAddCyclic.card_nsmul_eq_zero_le]
theorem IsCyclic.card_pow_eq_one_le [DecidableEq α] [Fintype α] [IsCyclic α] {n : ℕ} (hn0 : 0 < n) :
    #{a : α | a ^ n = 1} ≤ n :=
  let ⟨g, hg⟩ := IsCyclic.exists_generator (α := α)
  calc
    #{a : α | a ^ n = 1} ≤
        #(zpowers (g ^ (Fintype.card α / Nat.gcd n (Fintype.card α))) : Set α).toFinset :=
      card_le_card fun x hx =>
        let ⟨m, hm⟩ := show x ∈ Submonoid.powers g from mem_powers_iff_mem_zpowers.2 <| hg x
        Set.mem_toFinset.2
          ⟨(m / (Fintype.card α / Nat.gcd n (Fintype.card α)) : ℕ), by
            /-
              α : Type u_1
              inst✝³ : Group α
              inst✝² : DecidableEq α
              inst✝¹ : Fintype α
              inst✝ : IsCyclic α
              n : Nat
              hn0 : LT.lt 0 n
              g : α
              hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
              x : α
              hx : Membership.mem (Finset.filter (fun a => Eq (HPow.hPow a n) 1) Finset.univ …
              m : Nat
              hm : Eq ((fun x => HPow.hPow g x) m) x
              ⊢ Eq ((fun x => HPow.hPow (HPow.hPow g (HDiv.hDiv (Fintype.card α) (n.gcd (Fin …
            -/
            dsimp at hm
            have hgmn : g ^ (m * Nat.gcd n (Fintype.card α)) = 1 := by
              rw [pow_mul, hm, ← pow_gcd_card_eq_one_iff]; exact (mem_filter.1 hx).2
            /-
              α : Type u_1
              inst✝³ : Group α
              inst✝² : DecidableEq α
              inst✝¹ : Fintype α
              inst✝ : IsCyclic α
              n : Nat
              hn0 : LT.lt 0 n
              g : α
              hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
              x : α
              hx : Membership.mem (Finset.filter (fun a => Eq (HPow.hPow a n) 1) Finset.univ …
              m : Nat
              hm : Eq (HPow.hPow g m) x
              hgmn : Eq (HPow.hPow g (HMul.hMul m (n.gcd (Fintype.card α)))) 1
              ⊢ Eq ((fun x => HPow.hPow (HPow.hPow g (HDiv.hDiv (Fintype.card α) (n.gcd (Fin …
            -/
            dsimp only
            /-
              α : Type u_1
              inst✝³ : Group α
              inst✝² : DecidableEq α
              inst✝¹ : Fintype α
              inst✝ : IsCyclic α
              n : Nat
              hn0 : LT.lt 0 n
              g : α
              hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
              x : α
              hx : Membership.mem (Finset.filter (fun a => Eq (HPow.hPow a n) 1) Finset.univ …
              m : Nat
              hm : Eq (HPow.hPow g m) x
              hgmn : Eq (HPow.hPow g (HMul.hMul m (n.gcd (Fintype.card α)))) 1
              ⊢ Eq (HPow.hPow (HPow.hPow g (HDiv.hDiv (Fintype.card α) (n.gcd (Fintype.card  …
            -/
            rw [zpow_natCast, ← pow_mul, Nat.mul_div_cancel_left', hm]
            /-
              α : Type u_1
              inst✝³ : Group α
              inst✝² : DecidableEq α
              inst✝¹ : Fintype α
              inst✝ : IsCyclic α
              n : Nat
              hn0 : LT.lt 0 n
              g : α
              hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
              x : α
              hx : Membership.mem (Finset.filter (fun a => Eq (HPow.hPow a n) 1) Finset.univ …
              m : Nat
              hm : Eq (HPow.hPow g m) x
              hgmn : Eq (HPow.hPow g (HMul.hMul m (n.gcd (Fintype.card α)))) 1
              ⊢ Dvd.dvd (HDiv.hDiv (Fintype.card α) (n.gcd (Fintype.card α))) m
            -/
            refine Nat.dvd_of_mul_dvd_mul_right (gcd_pos_of_pos_left (Fintype.card α) hn0) ?_
            conv_lhs =>
              rw [Nat.div_mul_cancel (Nat.gcd_dvd_right _ _), ← Nat.card_eq_fintype_card,
                ← orderOf_eq_card_of_forall_mem_zpowers hg]
            /-
              α : Type u_1
              inst✝³ : Group α
              inst✝² : DecidableEq α
              inst✝¹ : Fintype α
              inst✝ : IsCyclic α
              n : Nat
              hn0 : LT.lt 0 n
              g : α
              hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
              x : α
              hx : Membership.mem (Finset.filter (fun a => Eq (HPow.hPow a n) 1) Finset.univ …
              m : Nat
              hm : Eq (HPow.hPow g m) x
              hgmn : Eq (HPow.hPow g (HMul.hMul m (n.gcd (Fintype.card α)))) 1
              ⊢ Dvd.dvd (orderOf g) (HMul.hMul m (n.gcd (Fintype.card α)))
            -/
            exact orderOf_dvd_of_pow_eq_one hgmn⟩
            /-
              🎉 no goals
            -/
    _ ≤ n := by
      /-
        α : Type u_1
        inst✝³ : Group α
        inst✝² : DecidableEq α
        inst✝¹ : Fintype α
        inst✝ : IsCyclic α
        n : Nat
        hn0 : LT.lt 0 n
        g : α
        hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
        ⊢ LE.le (↑(Subgroup.zpowers (HPow.hPow g (HDiv.hDiv (Fintype.card α) (n.gcd (F …
      -/
      let ⟨m, hm⟩ := Nat.gcd_dvd_right n (Fintype.card α)
      have hm0 : 0 < m :=
        Nat.pos_of_ne_zero fun hm0 => by
          rw [hm0, mul_zero, Fintype.card_eq_zero_iff] at hm
          exact hm.elim' 1
      /-
        α : Type u_1
        inst✝³ : Group α
        inst✝² : DecidableEq α
        inst✝¹ : Fintype α
        inst✝ : IsCyclic α
        n : Nat
        hn0 : LT.lt 0 n
        g : α
        hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
        m : Nat
        hm : Eq (Fintype.card α) (HMul.hMul (n.gcd (Fintype.card α)) m)
        hm0 : LT.lt 0 m
        ⊢ LE.le (↑(Subgroup.zpowers (HPow.hPow g (HDiv.hDiv (Fintype.card α) (n.gcd (F …
      -/
      simp only [Set.toFinset_card, SetLike.coe_sort_coe]
      rw [Fintype.card_zpowers, orderOf_pow g, orderOf_eq_card_of_forall_mem_zpowers hg,
        Nat.card_eq_fintype_card]
      /-
        α : Type u_1
        inst✝³ : Group α
        inst✝² : DecidableEq α
        inst✝¹ : Fintype α
        inst✝ : IsCyclic α
        n : Nat
        hn0 : LT.lt 0 n
        g : α
        hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
        m : Nat
        hm : Eq (Fintype.card α) (HMul.hMul (n.gcd (Fintype.card α)) m)
        hm0 : LT.lt 0 m
        ⊢ LE.le (HDiv.hDiv (Fintype.card α) ((Fintype.card α).gcd (HDiv.hDiv (Fintype. …
      -/
      nth_rw 2 [hm]; nth_rw 3 [hm]
      rw [Nat.mul_div_cancel_left _ (gcd_pos_of_pos_left _ hn0), gcd_mul_left_left, hm,
        Nat.mul_div_cancel _ hm0]
      /-
        α : Type u_1
        inst✝³ : Group α
        inst✝² : DecidableEq α
        inst✝¹ : Fintype α
        inst✝ : IsCyclic α
        n : Nat
        hn0 : LT.lt 0 n
        g : α
        hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
        m : Nat
        hm : Eq (Fintype.card α) (HMul.hMul (n.gcd (Fintype.card α)) m)
        hm0 : LT.lt 0 m
        ⊢ LE.le (n.gcd (Fintype.card α)) n
      -/
      exact le_of_dvd hn0 (Nat.gcd_dvd_left _ _)
      /-
        🎉 no goals
      -/

@[deprecated (since := "2024-02-21")]
alias IsAddCyclic.card_pow_eq_one_le := IsAddCyclic.card_nsmul_eq_zero_le


@[to_additive]
theorem IsCyclic.exists_monoid_generator [Finite α] [IsCyclic α] :
    ∃ x : α, ∀ y : α, y ∈ Submonoid.powers x := by
  /-
    α : Type u_1
    inst✝² : Group α
    inst✝¹ : Finite α
    inst✝ : IsCyclic α
    ⊢ Exists fun x => ∀ (y : α), Membership.mem (Submonoid.powers x) y
  -/
  simp_rw [mem_powers_iff_mem_zpowers]
  /-
    α : Type u_1
    inst✝² : Group α
    inst✝¹ : Finite α
    inst✝ : IsCyclic α
    ⊢ Exists fun x => ∀ (y : α), Membership.mem (Subgroup.zpowers x) y
  -/
  exact IsCyclic.exists_generator
  /-
    🎉 no goals
  -/


@[to_additive]
lemma IsCyclic.exists_ofOrder_eq_natCard [h : IsCyclic α] : ∃ g : α, orderOf g = Nat.card α := by
  /-
    α : Type u_1
    inst✝ : Group α
    h : IsCyclic α
    ⊢ Exists fun g => Eq (orderOf g) (Nat.card α)
  -/
  obtain ⟨g, hg⟩ := h.exists_generator
  /-
    case intro
    α : Type u_1
    inst✝ : Group α
    h : IsCyclic α
    g : α
    hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
    ⊢ Exists fun g => Eq (orderOf g) (Nat.card α)
  -/
  use g
  /-
    case h
    α : Type u_1
    inst✝ : Group α
    h : IsCyclic α
    g : α
    hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
    ⊢ Eq (orderOf g) (Nat.card α)
  -/
  rw [← card_zpowers g, (eq_top_iff' (zpowers g)).mpr hg]
  /-
    case h
    α : Type u_1
    inst✝ : Group α
    h : IsCyclic α
    g : α
    hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
    ⊢ Eq (Nat.card (Subtype fun x => Membership.mem Top.top x)) (Nat.card α)
  -/
  exact Nat.card_congr (Equiv.Set.univ α)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsCyclic.unique_zpow_zmod (ha : ∀ x : α, x ∈ zpowers a) (x : α) :
    ∃! n : ZMod (Fintype.card α), x = a ^ n.val := by
  /-
    α : Type u_1
    a : α
    inst✝¹ : Group α
    inst✝ : Fintype α
    ha : ∀ (x : α), Membership.mem (Subgroup.zpowers a) x
    x : α
    ⊢ ExistsUnique fun n => Eq x (HPow.hPow a n.val)
  -/
  obtain ⟨n, rfl⟩ := ha x
  /-
    case intro
    α : Type u_1
    a : α
    inst✝¹ : Group α
    inst✝ : Fintype α
    ha : ∀ (x : α), Membership.mem (Subgroup.zpowers a) x
    n : Int
    ⊢ ExistsUnique fun n_1 => Eq ((fun x => HPow.hPow a x) n) (HPow.hPow a n_1.val)
  -/
  refine ⟨n, (?_ : a ^ n = _), fun y (hy : a ^ n = _) ↦ ?_⟩
  · rw [← zpow_natCast, zpow_eq_zpow_iff_modEq, orderOf_eq_card_of_forall_mem_zpowers ha,
      Int.modEq_comm, Int.modEq_iff_add_fac, Nat.card_eq_fintype_card, ← ZMod.intCast_eq_iff]
  · rw [← zpow_natCast, zpow_eq_zpow_iff_modEq, orderOf_eq_card_of_forall_mem_zpowers ha,
      Nat.card_eq_fintype_card, ← ZMod.intCast_eq_intCast_iff] at hy
    /-
      case intro.refine_2
      α : Type u_1
      a : α
      inst✝¹ : Group α
      inst✝ : Fintype α
      ha : ∀ (x : α), Membership.mem (Subgroup.zpowers a) x
      n : Int
      y : ZMod (Fintype.card α)
      hy : Eq ↑n ↑↑y.val
      ⊢ Eq y ↑n
    -/
    simp [hy]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem IsCyclic.image_range_orderOf (ha : ∀ x : α, x ∈ zpowers a) :
    Finset.image (fun i => a ^ i) (range (orderOf a)) = univ := by
  /-
    α : Type u_1
    a : α
    inst✝² : Group α
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    ha : ∀ (x : α), Membership.mem (Subgroup.zpowers a) x
    ⊢ Eq (Finset.image (fun i => HPow.hPow a i) (Finset.range (orderOf a))) Finset …
  -/
  simp_rw [← SetLike.mem_coe] at ha
  /-
    α : Type u_1
    a : α
    inst✝² : Group α
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    ha : ∀ (x : α), Membership.mem (↑(Subgroup.zpowers a)) x
    ⊢ Eq (Finset.image (fun i => HPow.hPow a i) (Finset.range (orderOf a))) Finset …
  -/
  simp only [_root_.image_range_orderOf, Set.eq_univ_iff_forall.mpr ha, Set.toFinset_univ]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsCyclic.image_range_card (ha : ∀ x : α, x ∈ zpowers a) :
    Finset.image (fun i => a ^ i) (range (Nat.card α)) = univ := by
  /-
    α : Type u_1
    a : α
    inst✝² : Group α
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    ha : ∀ (x : α), Membership.mem (Subgroup.zpowers a) x
    ⊢ Eq (Finset.image (fun i => HPow.hPow a i) (Finset.range (Nat.card α))) Finse …
  -/
  rw [← orderOf_eq_card_of_forall_mem_zpowers ha, IsCyclic.image_range_orderOf ha]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma IsCyclic.ext [Finite G] [IsCyclic G] {d : ℕ} {a b : ZMod d}
    (hGcard : Nat.card G = d) (h : ∀ t : G, t ^ a.val = t ^ b.val) : a = b := by
  /-
    G : Type u_2
    inst✝² : Group G
    inst✝¹ : Finite G
    inst✝ : IsCyclic G
    d : Nat
    a b : ZMod d
    hGcard : Eq (Nat.card G) d
    h : ∀ (t : G), Eq (HPow.hPow t a.val) (HPow.hPow t b.val)
    ⊢ Eq a b
  -/
  have : NeZero (Nat.card G) := ⟨Nat.card_pos.ne'⟩
  /-
    G : Type u_2
    inst✝² : Group G
    inst✝¹ : Finite G
    inst✝ : IsCyclic G
    d : Nat
    a b : ZMod d
    hGcard : Eq (Nat.card G) d
    h : ∀ (t : G), Eq (HPow.hPow t a.val) (HPow.hPow t b.val)
    this : NeZero (Nat.card G)
    ⊢ Eq a b
  -/
  obtain ⟨g, hg⟩ := IsCyclic.exists_generator (α := G)
  /-
    case intro
    G : Type u_2
    inst✝² : Group G
    inst✝¹ : Finite G
    inst✝ : IsCyclic G
    d : Nat
    a b : ZMod d
    hGcard : Eq (Nat.card G) d
    h : ∀ (t : G), Eq (HPow.hPow t a.val) (HPow.hPow t b.val)
    this : NeZero (Nat.card G)
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    ⊢ Eq a b
  -/
  specialize h g
  /-
    case intro
    G : Type u_2
    inst✝² : Group G
    inst✝¹ : Finite G
    inst✝ : IsCyclic G
    d : Nat
    a b : ZMod d
    hGcard : Eq (Nat.card G) d
    this : NeZero (Nat.card G)
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    h : Eq (HPow.hPow g a.val) (HPow.hPow g b.val)
    ⊢ Eq a b
  -/
  subst hGcard
  rw [pow_eq_pow_iff_modEq, orderOf_eq_card_of_forall_mem_zpowers hg,
    ← ZMod.natCast_eq_natCast_iff] at h
  /-
    case intro
    G : Type u_2
    inst✝² : Group G
    inst✝¹ : Finite G
    inst✝ : IsCyclic G
    this : NeZero (Nat.card G)
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    a b : ZMod (Nat.card G)
    h : Eq ↑a.val ↑b.val
    ⊢ Eq a b
  -/
  simpa [ZMod.natCast_val, ZMod.cast_id'] using h
  /-
    🎉 no goals
  -/


@[to_additive]
private theorem card_pow_eq_one_eq_orderOf_aux (a : α) : #{b : α | b ^ orderOf a = 1} = orderOf a :=
  le_antisymm (hn _ (orderOf_pos a))
    (calc
      orderOf a = @Fintype.card (zpowers a) (id _) := Fintype.card_zpowers.symm
      _ ≤
          @Fintype.card (({b : α | b ^ orderOf a = 1} : Finset _) : Set α)
            (Fintype.ofFinset _ fun _ => Iff.rfl) :=
        (@Fintype.card_le_of_injective (zpowers a)
          (({b : α | b ^ orderOf a = 1} : Finset _) : Set α) (id _) (id _)
          (fun b =>
            ⟨b.1,
              mem_filter.2
                ⟨mem_univ _, by
                  /-
                    α : Type u_1
                    inst✝² : Group α
                    inst✝¹ : DecidableEq α
                    inst✝ : Fintype α
                    hn : ∀ (n : Nat), LT.lt 0 n → LE.le (Finset.filter (fun a => Eq (HPow.hPow a n …
                    a : α
                    b : Subtype fun x => Membership.mem (Subgroup.zpowers a) x
                    ⊢ Eq (HPow.hPow (↑b) (orderOf a)) 1
                  -/
                  let ⟨i, hi⟩ := b.2
                  rw [← hi, ← zpow_natCast, ← zpow_mul, mul_comm, zpow_mul, zpow_natCast,
                    pow_orderOf_eq_one, one_zpow]⟩⟩)
          fun _ _ h => Subtype.eq (Subtype.mk.inj h))
      _ = #{b : α | b ^ orderOf a = 1} := Fintype.card_ofFinset _ _
      )

-- Use φ for `Nat.totient`

@[to_additive]
private theorem card_orderOf_eq_totient_aux₁ {d : ℕ} (hd : d ∣ Fintype.card α)
    (hpos : 0 < #{a : α | orderOf a = d}) : #{a : α | orderOf a = d} = φ d := by
  /-
    α : Type u_1
    inst✝² : Group α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hn : ∀ (n : Nat), LT.lt 0 n → LE.le (Finset.filter (fun a => Eq (HPow.hPow a n …
    d : Nat
    hd : Dvd.dvd d (Fintype.card α)
    hpos : LT.lt 0 (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ).card
    ⊢ Eq (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ).card d.totient
  -/
  induction' d using Nat.strongRec' with d IH
  /-
    case H
    α : Type u_1
    inst✝² : Group α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hn : ∀ (n : Nat), LT.lt 0 n → LE.le (Finset.filter (fun a => Eq (HPow.hPow a n …
    d : Nat
    IH : ∀ (m : Nat), LT.lt m d → Dvd.dvd m (Fintype.card α) → LT.lt 0 (Finset.fil …
    hd : Dvd.dvd d (Fintype.card α)
    hpos : LT.lt 0 (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ).card
    ⊢ Eq (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ).card d.totient
  -/
  rcases Decidable.eq_or_ne d 0 with (rfl | hd0)
    /-
      case H.inl
      α : Type u_1
      inst✝² : Group α
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      hn : ∀ (n : Nat), LT.lt 0 n → LE.le (Finset.filter (fun a => Eq (HPow.hPow a n …
      IH : ∀ (m : Nat), LT.lt m 0 → Dvd.dvd m (Fintype.card α) → LT.lt 0 (Finset.fil …
      hd : Dvd.dvd 0 (Fintype.card α)
      hpos : LT.lt 0 (Finset.filter (fun a => Eq (orderOf a) 0) Finset.univ).card
      ⊢ Eq (Finset.filter (fun a => Eq (orderOf a) 0) Finset.univ).card (Nat.totient …
    -/
  · cases Fintype.card_ne_zero (eq_zero_of_zero_dvd hd)
    /-
      🎉 no goals
    -/
  /-
    case H.inr
    α : Type u_1
    inst✝² : Group α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hn : ∀ (n : Nat), LT.lt 0 n → LE.le (Finset.filter (fun a => Eq (HPow.hPow a n …
    d : Nat
    IH : ∀ (m : Nat), LT.lt m d → Dvd.dvd m (Fintype.card α) → LT.lt 0 (Finset.fil …
    hd : Dvd.dvd d (Fintype.card α)
    hpos : LT.lt 0 (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ).card
    hd0 : Ne d 0
    ⊢ Eq (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ).card d.totient
  -/
  rcases Finset.card_pos.1 hpos with ⟨a, ha'⟩
  /-
    case H.inr.intro
    α : Type u_1
    inst✝² : Group α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hn : ∀ (n : Nat), LT.lt 0 n → LE.le (Finset.filter (fun a => Eq (HPow.hPow a n …
    d : Nat
    IH : ∀ (m : Nat), LT.lt m d → Dvd.dvd m (Fintype.card α) → LT.lt 0 (Finset.fil …
    hd : Dvd.dvd d (Fintype.card α)
    hpos : LT.lt 0 (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ).card
    hd0 : Ne d 0
    a : α
    ha' : Membership.mem (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ) a
    ⊢ Eq (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ).card d.totient
  -/
  have ha : orderOf a = d := (mem_filter.1 ha').2
  have h1 :
    (∑ m ∈ d.properDivisors, #{a : α | orderOf a = m}) =
      ∑ m ∈ d.properDivisors, φ m := by
    refine Finset.sum_congr rfl fun m hm => ?_
    simp only [mem_filter, mem_range, mem_properDivisors] at hm
    refine IH m hm.2 (hm.1.trans hd) (Finset.card_pos.2 ⟨a ^ (d / m), ?_⟩)
    simp only [mem_filter, mem_univ, orderOf_pow a, ha, true_and,
      Nat.gcd_eq_right (div_dvd_of_dvd hm.1), Nat.div_div_self hm.1 hd0]
  have h2 :
    (∑ m ∈ d.divisors, #{a : α | orderOf a = m}) =
      ∑ m ∈ d.divisors, φ m := by
    rw [← filter_dvd_eq_divisors hd0, sum_card_orderOf_eq_card_pow_eq_one hd0,
      filter_dvd_eq_divisors hd0, sum_totient, ← ha, card_pow_eq_one_eq_orderOf_aux hn a]
  /-
    case H.inr.intro
    α : Type u_1
    inst✝² : Group α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hn : ∀ (n : Nat), LT.lt 0 n → LE.le (Finset.filter (fun a => Eq (HPow.hPow a n …
    d : Nat
    IH : ∀ (m : Nat), LT.lt m d → Dvd.dvd m (Fintype.card α) → LT.lt 0 (Finset.fil …
    hd : Dvd.dvd d (Fintype.card α)
    hpos : LT.lt 0 (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ).card
    hd0 : Ne d 0
    a : α
    ha' : Membership.mem (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ) a
    ha : Eq (orderOf a) d
    h1 : Eq (d.properDivisors.sum fun m => (Finset.filter (fun a => Eq (orderOf a) …
    h2 : Eq (d.divisors.sum fun m => (Finset.filter (fun a => Eq (orderOf a) m) Fi …
    ⊢ Eq (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ).card d.totient
  -/
  simpa [← cons_self_properDivisors hd0, ← h1] using h2
  /-
    🎉 no goals
  -/


@[to_additive]
theorem card_orderOf_eq_totient_aux₂ {d : ℕ} (hd : d ∣ Fintype.card α) :
    #{a : α | orderOf a = d} = φ d := by
  /-
    α : Type u_1
    inst✝² : Group α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hn : ∀ (n : Nat), LT.lt 0 n → LE.le (Finset.filter (fun a => Eq (HPow.hPow a n …
    d : Nat
    hd : Dvd.dvd d (Fintype.card α)
    ⊢ Eq (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ).card d.totient
  -/
  let c := Fintype.card α
  /-
    α : Type u_1
    inst✝² : Group α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hn : ∀ (n : Nat), LT.lt 0 n → LE.le (Finset.filter (fun a => Eq (HPow.hPow a n …
    d : Nat
    hd : Dvd.dvd d (Fintype.card α)
    c : Nat := Fintype.card α
    ⊢ Eq (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ).card d.totient
  -/
  have hc0 : 0 < c := Fintype.card_pos_iff.2 ⟨1⟩
  /-
    α : Type u_1
    inst✝² : Group α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hn : ∀ (n : Nat), LT.lt 0 n → LE.le (Finset.filter (fun a => Eq (HPow.hPow a n …
    d : Nat
    hd : Dvd.dvd d (Fintype.card α)
    c : Nat := Fintype.card α
    hc0 : LT.lt 0 c
    ⊢ Eq (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ).card d.totient
  -/
  apply card_orderOf_eq_totient_aux₁ hn hd
  /-
    α : Type u_1
    inst✝² : Group α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hn : ∀ (n : Nat), LT.lt 0 n → LE.le (Finset.filter (fun a => Eq (HPow.hPow a n …
    d : Nat
    hd : Dvd.dvd d (Fintype.card α)
    c : Nat := Fintype.card α
    hc0 : LT.lt 0 c
    ⊢ LT.lt 0 (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ).card
  -/
  by_contra h0
  -- Must qualify `Finset.card_eq_zero` because of https://github.com/leanprover/lean4/issues/2849
  /-
    α : Type u_1
    inst✝² : Group α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hn : ∀ (n : Nat), LT.lt 0 n → LE.le (Finset.filter (fun a => Eq (HPow.hPow a n …
    d : Nat
    hd : Dvd.dvd d (Fintype.card α)
    c : Nat := Fintype.card α
    hc0 : LT.lt 0 c
    h0 : Not (LT.lt 0 (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ).card)
    ⊢ False
  -/
  simp_rw [not_lt, Nat.le_zero, Finset.card_eq_zero] at h0
  /-
    α : Type u_1
    inst✝² : Group α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hn : ∀ (n : Nat), LT.lt 0 n → LE.le (Finset.filter (fun a => Eq (HPow.hPow a n …
    d : Nat
    hd : Dvd.dvd d (Fintype.card α)
    c : Nat := Fintype.card α
    hc0 : LT.lt 0 c
    h0 : Eq (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ) EmptyCollectio …
    ⊢ False
  -/
  apply lt_irrefl c
  calc
    c = ∑ m ∈ c.divisors, #{a : α | orderOf a = m} := by
      simp only [← filter_dvd_eq_divisors hc0.ne', sum_card_orderOf_eq_card_pow_eq_one hc0.ne']
      apply congr_arg card
      simp [c]
    _ = ∑ m ∈ c.divisors.erase d, #{a : α | orderOf a = m} := by
      rw [eq_comm]
      refine sum_subset (erase_subset _ _) fun m hm₁ hm₂ => ?_
      have : m = d := by
        contrapose! hm₂
        exact mem_erase_of_ne_of_mem hm₂ hm₁
      simp [this, h0]
    _ ≤ ∑ m ∈ c.divisors.erase d, φ m := by
      refine sum_le_sum fun m hm => ?_
      have hmc : m ∣ c := by
        simp only [mem_erase, mem_divisors] at hm
        tauto
      obtain h1 | h1 := (#{a : α | orderOf a = m}).eq_zero_or_pos
      · simp [h1]
      · simp [card_orderOf_eq_totient_aux₁ hn hmc h1]
    _ < ∑ m ∈ c.divisors, φ m :=
      sum_erase_lt_of_pos (mem_divisors.2 ⟨hd, hc0.ne'⟩) (totient_pos.2 (pos_of_dvd_of_pos hd hc0))
    _ = c := sum_totient _


@[to_additive isAddCyclic_of_card_nsmul_eq_zero_le, stacks 09HX "This theorem is stronger than \
09HX. It removes the abelian condition, and requires only `≤` instead of `=`."]
theorem isCyclic_of_card_pow_eq_one_le : IsCyclic α :=
  have : Finset.Nonempty {a : α | orderOf a = Nat.card α} :=
    card_pos.1 <| by
      /-
        α : Type u_1
        inst✝² : Group α
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        hn : ∀ (n : Nat), LT.lt 0 n → LE.le (Finset.filter (fun a => Eq (HPow.hPow a n …
        ⊢ LT.lt 0 (Finset.filter (fun a => Eq (orderOf a) (Nat.card α)) Finset.univ).c …
      -/
      rw [Nat.card_eq_fintype_card, card_orderOf_eq_totient_aux₂ hn dvd_rfl, totient_pos]
      /-
        α : Type u_1
        inst✝² : Group α
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        hn : ∀ (n : Nat), LT.lt 0 n → LE.le (Finset.filter (fun a => Eq (HPow.hPow a n …
        ⊢ LT.lt 0 (Fintype.card α)
      -/
      apply Fintype.card_pos
      /-
        🎉 no goals
      -/
  let ⟨x, hx⟩ := this
  isCyclic_of_orderOf_eq_card x (Finset.mem_filter.1 hx).2


@[deprecated (since := "2024-02-21")]
alias isAddCyclic_of_card_pow_eq_one_le := isAddCyclic_of_card_nsmul_eq_zero_le


@[to_additive]
lemma IsCyclic.card_orderOf_eq_totient [IsCyclic α] [Fintype α] {d : ℕ} (hd : d ∣ Fintype.card α) :
    #{a : α | orderOf a = d} = totient d := by
  /-
    α : Type u_1
    inst✝² : Group α
    inst✝¹ : IsCyclic α
    inst✝ : Fintype α
    d : Nat
    hd : Dvd.dvd d (Fintype.card α)
    ⊢ Eq (Finset.filter (fun a => Eq (orderOf a) d) Finset.univ).card d.totient
  -/
  classical apply card_orderOf_eq_totient_aux₂ (fun n => IsCyclic.card_pow_eq_one_le) hd
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-21")]
alias IsAddCyclic.card_orderOf_eq_totient := IsAddCyclic.card_addOrderOf_eq_totient


/-- A finite group of prime order is simple. -/
@[to_additive "A finite group of prime order is simple."]
theorem isSimpleGroup_of_prime_card {p : ℕ} [hp : Fact p.Prime]
    (h : Nat.card α = p) : IsSimpleGroup α := by
  /-
    α : Type u_1
    inst✝ : Group α
    p : Nat
    hp : Fact (Nat.Prime p)
    h : Eq (Nat.card α) p
    ⊢ IsSimpleGroup α
  -/
  subst h
  /-
    α : Type u_1
    inst✝ : Group α
    hp : Fact (Nat.Prime (Nat.card α))
    ⊢ IsSimpleGroup α
  -/
  have : Finite α := Nat.finite_of_card_ne_zero hp.1.ne_zero
  /-
    α : Type u_1
    inst✝ : Group α
    hp : Fact (Nat.Prime (Nat.card α))
    this : Finite α
    ⊢ IsSimpleGroup α
  -/
  have : Nontrivial α := Finite.one_lt_card_iff_nontrivial.mp hp.1.one_lt
  /-
    α : Type u_1
    inst✝ : Group α
    hp : Fact (Nat.Prime (Nat.card α))
    this✝ : Finite α
    this : Nontrivial α
    ⊢ IsSimpleGroup α
  -/
  exact ⟨fun H _ => H.eq_bot_or_eq_top_of_prime_card⟩
  /-
    🎉 no goals
  -/


/-- A group is commutative if the quotient by the center is cyclic.
  Also see `commGroupOfCyclicCenterQuotient` for the `CommGroup` instance. -/
@[to_additive
      "A group is commutative if the quotient by the center is cyclic.
      Also see `addCommGroupOfCyclicCenterQuotient` for the `AddCommGroup` instance."]
theorem commutative_of_cyclic_center_quotient [IsCyclic G'] (f : G →* G') (hf : f.ker ≤ center G)
    (a b : G) : a * b = b * a :=
  let ⟨⟨x, y, (hxy : f y = x)⟩, (hx : ∀ a : f.range, a ∈ zpowers _)⟩ :=
    IsCyclic.exists_generator (α := f.range)
  let ⟨m, hm⟩ := hx ⟨f a, a, rfl⟩
  let ⟨n, hn⟩ := hx ⟨f b, b, rfl⟩
                              /-
                                G : Type u_2
                                G' : Type u_3
                                inst✝² : Group G
                                inst✝¹ : Group G'
                                inst✝ : IsCyclic G'
                                f : MonoidHom G G'
                                hf : LE.le f.ker (Subgroup.center G)
                                a b : G
                                x : G'
                                y : G
                                hxy : Eq (f y) x
                                hx : ∀ (a : Subtype fun x => Membership.mem f.range x), Membership.mem (Subgro …
                                m : Int
                                hm : Eq ((fun x_1 => HPow.hPow ⟨x, ⋯⟩ x_1) m) ⟨f a, ⋯⟩
                                n : Int
                                hn : Eq ((fun x_1 => HPow.hPow ⟨x, ⋯⟩ x_1) n) ⟨f b, ⋯⟩
                                ⊢ Eq (HPow.hPow x m) (f a)
                              -/
  have hm : x ^ m = f a := by simpa [Subtype.ext_iff] using hm
                              /-
                                🎉 no goals
                              -/
                              /-
                                G : Type u_2
                                G' : Type u_3
                                inst✝² : Group G
                                inst✝¹ : Group G'
                                inst✝ : IsCyclic G'
                                f : MonoidHom G G'
                                hf : LE.le f.ker (Subgroup.center G)
                                a b : G
                                x : G'
                                y : G
                                hxy : Eq (f y) x
                                hx : ∀ (a : Subtype fun x => Membership.mem f.range x), Membership.mem (Subgro …
                                m : Int
                                hm✝ : Eq ((fun x_1 => HPow.hPow ⟨x, ⋯⟩ x_1) m) ⟨f a, ⋯⟩
                                n : Int
                                hn : Eq ((fun x_1 => HPow.hPow ⟨x, ⋯⟩ x_1) n) ⟨f b, ⋯⟩
                                hm : Eq (HPow.hPow x m) (f a)
                                ⊢ Eq (HPow.hPow x n) (f b)
                              -/
  have hn : x ^ n = f b := by simpa [Subtype.ext_iff] using hn
                              /-
                                🎉 no goals
                              -/
  have ha : y ^ (-m) * a ∈ center G :=
           /-
             G : Type u_2
             G' : Type u_3
             inst✝² : Group G
             inst✝¹ : Group G'
             inst✝ : IsCyclic G'
             f : MonoidHom G G'
             hf : LE.le f.ker (Subgroup.center G)
             a b : G
             x : G'
             y : G
             hxy : Eq (f y) x
             hx : ∀ (a : Subtype fun x => Membership.mem f.range x), Membership.mem (Subgro …
             m : Int
             hm✝ : Eq ((fun x_1 => HPow.hPow ⟨x, ⋯⟩ x_1) m) ⟨f a, ⋯⟩
             n : Int
             hn✝ : Eq ((fun x_1 => HPow.hPow ⟨x, ⋯⟩ x_1) n) ⟨f b, ⋯⟩
             hm : Eq (HPow.hPow x m) (f a)
             hn : Eq (HPow.hPow x n) (f b)
             ⊢ Membership.mem f.ker (HMul.hMul (HPow.hPow y (Neg.neg m)) a)
           -/
    hf (by rw [f.mem_ker, f.map_mul, f.map_zpow, hxy, zpow_neg x m, hm, inv_mul_cancel])
           /-
             🎉 no goals
           -/
  have hb : y ^ (-n) * b ∈ center G :=
           /-
             G : Type u_2
             G' : Type u_3
             inst✝² : Group G
             inst✝¹ : Group G'
             inst✝ : IsCyclic G'
             f : MonoidHom G G'
             hf : LE.le f.ker (Subgroup.center G)
             a b : G
             x : G'
             y : G
             hxy : Eq (f y) x
             hx : ∀ (a : Subtype fun x => Membership.mem f.range x), Membership.mem (Subgro …
             m : Int
             hm✝ : Eq ((fun x_1 => HPow.hPow ⟨x, ⋯⟩ x_1) m) ⟨f a, ⋯⟩
             n : Int
             hn✝ : Eq ((fun x_1 => HPow.hPow ⟨x, ⋯⟩ x_1) n) ⟨f b, ⋯⟩
             hm : Eq (HPow.hPow x m) (f a)
             hn : Eq (HPow.hPow x n) (f b)
             ha : Membership.mem (Subgroup.center G) (HMul.hMul (HPow.hPow y (Neg.neg m)) a)
             ⊢ Membership.mem f.ker (HMul.hMul (HPow.hPow y (Neg.neg n)) b)
           -/
    hf (by rw [f.mem_ker, f.map_mul, f.map_zpow, hxy, zpow_neg x n, hn, inv_mul_cancel])
           /-
             🎉 no goals
           -/
  calc
                                                                  /-
                                                                    G : Type u_2
                                                                    G' : Type u_3
                                                                    inst✝² : Group G
                                                                    inst✝¹ : Group G'
                                                                    inst✝ : IsCyclic G'
                                                                    f : MonoidHom G G'
                                                                    hf : LE.le f.ker (Subgroup.center G)
                                                                    a b : G
                                                                    x : G'
                                                                    y : G
                                                                    hxy : Eq (f y) x
                                                                    hx : ∀ (a : Subtype fun x => Membership.mem f.range x), Membership.mem (Subgro …
                                                                    m : Int
                                                                    hm✝ : Eq ((fun x_1 => HPow.hPow ⟨x, ⋯⟩ x_1) m) ⟨f a, ⋯⟩
                                                                    n : Int
                                                                    hn✝ : Eq ((fun x_1 => HPow.hPow ⟨x, ⋯⟩ x_1) n) ⟨f b, ⋯⟩
                                                                    hm : Eq (HPow.hPow x m) (f a)
                                                                    hn : Eq (HPow.hPow x n) (f b)
                                                                    ha : Membership.mem (Subgroup.center G) (HMul.hMul (HPow.hPow y (Neg.neg m)) a)
                                                                    hb : Membership.mem (Subgroup.center G) (HMul.hMul (HPow.hPow y (Neg.neg n)) b)
                                                                    ⊢ Eq (HMul.hMul a b) (HMul.hMul (HMul.hMul (HPow.hPow y m) (HMul.hMul (HMul.hM …
                                                                  -/
    a * b = y ^ m * (y ^ (-m) * a * y ^ n) * (y ^ (-n) * b) := by simp [mul_assoc]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
                                                                /-
                                                                  G : Type u_2
                                                                  G' : Type u_3
                                                                  inst✝² : Group G
                                                                  inst✝¹ : Group G'
                                                                  inst✝ : IsCyclic G'
                                                                  f : MonoidHom G G'
                                                                  hf : LE.le f.ker (Subgroup.center G)
                                                                  a b : G
                                                                  x : G'
                                                                  y : G
                                                                  hxy : Eq (f y) x
                                                                  hx : ∀ (a : Subtype fun x => Membership.mem f.range x), Membership.mem (Subgro …
                                                                  m : Int
                                                                  hm✝ : Eq ((fun x_1 => HPow.hPow ⟨x, ⋯⟩ x_1) m) ⟨f a, ⋯⟩
                                                                  n : Int
                                                                  hn✝ : Eq ((fun x_1 => HPow.hPow ⟨x, ⋯⟩ x_1) n) ⟨f b, ⋯⟩
                                                                  hm : Eq (HPow.hPow x m) (f a)
                                                                  hn : Eq (HPow.hPow x n) (f b)
                                                                  ha : Membership.mem (Subgroup.center G) (HMul.hMul (HPow.hPow y (Neg.neg m)) a)
                                                                  hb : Membership.mem (Subgroup.center G) (HMul.hMul (HPow.hPow y (Neg.neg n)) b)
                                                                  ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow y m) (HMul.hMul (HMul.hMul (HPow.hPow y  …
                                                                -/
    _ = y ^ m * (y ^ n * (y ^ (-m) * a)) * (y ^ (-n) * b) := by rw [mem_center_iff.1 ha]
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                              /-
                                                                G : Type u_2
                                                                G' : Type u_3
                                                                inst✝² : Group G
                                                                inst✝¹ : Group G'
                                                                inst✝ : IsCyclic G'
                                                                f : MonoidHom G G'
                                                                hf : LE.le f.ker (Subgroup.center G)
                                                                a b : G
                                                                x : G'
                                                                y : G
                                                                hxy : Eq (f y) x
                                                                hx : ∀ (a : Subtype fun x => Membership.mem f.range x), Membership.mem (Subgro …
                                                                m : Int
                                                                hm✝ : Eq ((fun x_1 => HPow.hPow ⟨x, ⋯⟩ x_1) m) ⟨f a, ⋯⟩
                                                                n : Int
                                                                hn✝ : Eq ((fun x_1 => HPow.hPow ⟨x, ⋯⟩ x_1) n) ⟨f b, ⋯⟩
                                                                hm : Eq (HPow.hPow x m) (f a)
                                                                hn : Eq (HPow.hPow x n) (f b)
                                                                ha : Membership.mem (Subgroup.center G) (HMul.hMul (HPow.hPow y (Neg.neg m)) a)
                                                                hb : Membership.mem (Subgroup.center G) (HMul.hMul (HPow.hPow y (Neg.neg n)) b)
                                                                ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow y m) (HMul.hMul (HPow.hPow y n) (HMul.hM …
                                                              -/
    _ = y ^ m * y ^ n * y ^ (-m) * (a * (y ^ (-n) * b)) := by simp [mul_assoc]
                                                              /-
                                                                🎉 no goals
                                                              -/
                                                            /-
                                                              G : Type u_2
                                                              G' : Type u_3
                                                              inst✝² : Group G
                                                              inst✝¹ : Group G'
                                                              inst✝ : IsCyclic G'
                                                              f : MonoidHom G G'
                                                              hf : LE.le f.ker (Subgroup.center G)
                                                              a b : G
                                                              x : G'
                                                              y : G
                                                              hxy : Eq (f y) x
                                                              hx : ∀ (a : Subtype fun x => Membership.mem f.range x), Membership.mem (Subgro …
                                                              m : Int
                                                              hm✝ : Eq ((fun x_1 => HPow.hPow ⟨x, ⋯⟩ x_1) m) ⟨f a, ⋯⟩
                                                              n : Int
                                                              hn✝ : Eq ((fun x_1 => HPow.hPow ⟨x, ⋯⟩ x_1) n) ⟨f b, ⋯⟩
                                                              hm : Eq (HPow.hPow x m) (f a)
                                                              hn : Eq (HPow.hPow x n) (f b)
                                                              ha : Membership.mem (Subgroup.center G) (HMul.hMul (HPow.hPow y (Neg.neg m)) a)
                                                              hb : Membership.mem (Subgroup.center G) (HMul.hMul (HPow.hPow y (Neg.neg n)) b)
                                                              ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow y m) (HPow.hPow y n)) (HPow.h …
                                                            -/
    _ = y ^ m * y ^ n * y ^ (-m) * (y ^ (-n) * b * a) := by rw [mem_center_iff.1 hb]
                                                            /-
                                                              🎉 no goals
                                                            -/
                    /-
                      G : Type u_2
                      G' : Type u_3
                      inst✝² : Group G
                      inst✝¹ : Group G'
                      inst✝ : IsCyclic G'
                      f : MonoidHom G G'
                      hf : LE.le f.ker (Subgroup.center G)
                      a b : G
                      x : G'
                      y : G
                      hxy : Eq (f y) x
                      hx : ∀ (a : Subtype fun x => Membership.mem f.range x), Membership.mem (Subgro …
                      m : Int
                      hm✝ : Eq ((fun x_1 => HPow.hPow ⟨x, ⋯⟩ x_1) m) ⟨f a, ⋯⟩
                      n : Int
                      hn✝ : Eq ((fun x_1 => HPow.hPow ⟨x, ⋯⟩ x_1) n) ⟨f b, ⋯⟩
                      hm : Eq (HPow.hPow x m) (f a)
                      hn : Eq (HPow.hPow x n) (f b)
                      ha : Membership.mem (Subgroup.center G) (HMul.hMul (HPow.hPow y (Neg.neg m)) a)
                      hb : Membership.mem (Subgroup.center G) (HMul.hMul (HPow.hPow y (Neg.neg n)) b)
                      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow y m) (HPow.hPow y n)) (HPow.h …
                    -/
    _ = b * a := by group
                    /-
                      🎉 no goals
                    -/


@[deprecated (since := "2024-02-21")]
alias commutative_of_add_cyclic_center_quotient := commutative_of_addCyclic_center_quotient


/-- A group is commutative if the quotient by the center is cyclic. -/
@[to_additive
      "A group is commutative if the quotient by the center is cyclic."]
def commGroupOfCyclicCenterQuotient [IsCyclic G'] (f : G →* G') (hf : f.ker ≤ center G) :
    CommGroup G :=
                    /-
                      α : Type u_1
                      G : Type u_2
                      G' : Type u_3
                      a : α
                      inst✝² : Group G
                      inst✝¹ : Group G'
                      inst✝ : IsCyclic G'
                      f : MonoidHom G G'
                      hf : LE.le f.ker (Subgroup.center G)
                      ⊢ Group G
                    -/
  { show Group G by infer_instance with mul_comm := commutative_of_cyclic_center_quotient f hf }
                    /-
                      🎉 no goals
                    -/


@[to_additive]
instance (priority := 100) isCyclic : IsCyclic α := by
  /-
    α : Type u_1
    G : Type u_2
    G' : Type u_3
    a : α
    inst✝¹ : CommGroup α
    inst✝ : IsSimpleGroup α
    ⊢ IsCyclic α
  -/
  nontriviality α
  /-
    α : Type u_1
    G : Type u_2
    G' : Type u_3
    a : α
    inst✝² : CommGroup α
    inst✝¹ : IsSimpleGroup α
    inst✝ : Nontrivial α
    ⊢ IsCyclic α
  -/
  obtain ⟨g, hg⟩ := exists_ne (1 : α)
  have : Subgroup.zpowers g = ⊤ :=
    (eq_bot_or_eq_top (Subgroup.zpowers g)).resolve_left (Subgroup.zpowers_ne_bot.2 hg)
  /-
    case intro
    α : Type u_1
    G : Type u_2
    G' : Type u_3
    a : α
    inst✝² : CommGroup α
    inst✝¹ : IsSimpleGroup α
    inst✝ : Nontrivial α
    g : α
    hg : Ne g 1
    this : Eq (Subgroup.zpowers g) Top.top
    ⊢ IsCyclic α
  -/
  exact ⟨⟨g, (Subgroup.eq_top_iff' _).1 this⟩⟩
  /-
    🎉 no goals
  -/


@[to_additive]
theorem prime_card [Finite α] : (Nat.card α).Prime := by
  /-
    α : Type u_1
    inst✝² : CommGroup α
    inst✝¹ : IsSimpleGroup α
    inst✝ : Finite α
    ⊢ Nat.Prime (Nat.card α)
  -/
  have h0 : 0 < Nat.card α := Nat.card_pos
  /-
    α : Type u_1
    inst✝² : CommGroup α
    inst✝¹ : IsSimpleGroup α
    inst✝ : Finite α
    h0 : LT.lt 0 (Nat.card α)
    ⊢ Nat.Prime (Nat.card α)
  -/
  obtain ⟨g, hg⟩ := IsCyclic.exists_generator (α := α)
  /-
    case intro
    α : Type u_1
    inst✝² : CommGroup α
    inst✝¹ : IsSimpleGroup α
    inst✝ : Finite α
    h0 : LT.lt 0 (Nat.card α)
    g : α
    hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
    ⊢ Nat.Prime (Nat.card α)
  -/
  rw [Nat.prime_def]
  /-
    case intro
    α : Type u_1
    inst✝² : CommGroup α
    inst✝¹ : IsSimpleGroup α
    inst✝ : Finite α
    h0 : LT.lt 0 (Nat.card α)
    g : α
    hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
    ⊢ And (LE.le 2 (Nat.card α)) (∀ (m : Nat), Dvd.dvd m (Nat.card α) → Or (Eq m 1 …
  -/
  refine ⟨Finite.one_lt_card_iff_nontrivial.2 inferInstance, fun n hn => ?_⟩
  /-
    case intro
    α : Type u_1
    inst✝² : CommGroup α
    inst✝¹ : IsSimpleGroup α
    inst✝ : Finite α
    h0 : LT.lt 0 (Nat.card α)
    g : α
    hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
    n : Nat
    hn : Dvd.dvd n (Nat.card α)
    ⊢ Or (Eq n 1) (Eq n (Nat.card α))
  -/
  refine (IsSimpleOrder.eq_bot_or_eq_top (Subgroup.zpowers (g ^ n))).symm.imp ?_ ?_
    /-
      case intro.refine_1
      α : Type u_1
      inst✝² : CommGroup α
      inst✝¹ : IsSimpleGroup α
      inst✝ : Finite α
      h0 : LT.lt 0 (Nat.card α)
      g : α
      hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
      n : Nat
      hn : Dvd.dvd n (Nat.card α)
      ⊢ Eq (Subgroup.zpowers (HPow.hPow g n)) Top.top → Eq n 1
    -/
  · intro h
    /-
      case intro.refine_1
      α : Type u_1
      inst✝² : CommGroup α
      inst✝¹ : IsSimpleGroup α
      inst✝ : Finite α
      h0 : LT.lt 0 (Nat.card α)
      g : α
      hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
      n : Nat
      hn : Dvd.dvd n (Nat.card α)
      h : Eq (Subgroup.zpowers (HPow.hPow g n)) Top.top
      ⊢ Eq n 1
    -/
    have hgo := orderOf_pow (n := n) g
    rw [orderOf_eq_card_of_forall_mem_zpowers hg, Nat.gcd_eq_right_iff_dvd.1 hn,
      orderOf_eq_card_of_forall_mem_zpowers, eq_comm,
      Nat.div_eq_iff_eq_mul_left (Nat.pos_of_dvd_of_pos hn h0) hn] at hgo
      /-
        case intro.refine_1
        α : Type u_1
        inst✝² : CommGroup α
        inst✝¹ : IsSimpleGroup α
        inst✝ : Finite α
        h0 : LT.lt 0 (Nat.card α)
        g : α
        hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
        n : Nat
        hn : Dvd.dvd n (Nat.card α)
        h : Eq (Subgroup.zpowers (HPow.hPow g n)) Top.top
        hgo : Eq (Nat.card α) (HMul.hMul (Nat.card α) n)
        ⊢ Eq n 1
      -/
    · exact (mul_left_cancel₀ (ne_of_gt h0) ((mul_one (Nat.card α)).trans hgo)).symm
      /-
        🎉 no goals
      -/
      /-
        case intro.refine_1
        α : Type u_1
        inst✝² : CommGroup α
        inst✝¹ : IsSimpleGroup α
        inst✝ : Finite α
        h0 : LT.lt 0 (Nat.card α)
        g : α
        hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
        n : Nat
        hn : Dvd.dvd n (Nat.card α)
        h : Eq (Subgroup.zpowers (HPow.hPow g n)) Top.top
        hgo : Eq (orderOf (HPow.hPow g n)) (HDiv.hDiv (Nat.card α) n)
        ⊢ ∀ (x : α), Membership.mem (Subgroup.zpowers (HPow.hPow g n)) x
      -/
    · intro x
      /-
        case intro.refine_1
        α : Type u_1
        inst✝² : CommGroup α
        inst✝¹ : IsSimpleGroup α
        inst✝ : Finite α
        h0 : LT.lt 0 (Nat.card α)
        g : α
        hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
        n : Nat
        hn : Dvd.dvd n (Nat.card α)
        h : Eq (Subgroup.zpowers (HPow.hPow g n)) Top.top
        hgo : Eq (orderOf (HPow.hPow g n)) (HDiv.hDiv (Nat.card α) n)
        x : α
        ⊢ Membership.mem (Subgroup.zpowers (HPow.hPow g n)) x
      -/
      rw [h]
      /-
        case intro.refine_1
        α : Type u_1
        inst✝² : CommGroup α
        inst✝¹ : IsSimpleGroup α
        inst✝ : Finite α
        h0 : LT.lt 0 (Nat.card α)
        g : α
        hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
        n : Nat
        hn : Dvd.dvd n (Nat.card α)
        h : Eq (Subgroup.zpowers (HPow.hPow g n)) Top.top
        hgo : Eq (orderOf (HPow.hPow g n)) (HDiv.hDiv (Nat.card α) n)
        x : α
        ⊢ Membership.mem Top.top x
      -/
      exact Subgroup.mem_top _
      /-
        🎉 no goals
      -/
    /-
      case intro.refine_2
      α : Type u_1
      inst✝² : CommGroup α
      inst✝¹ : IsSimpleGroup α
      inst✝ : Finite α
      h0 : LT.lt 0 (Nat.card α)
      g : α
      hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
      n : Nat
      hn : Dvd.dvd n (Nat.card α)
      ⊢ Eq (Subgroup.zpowers (HPow.hPow g n)) Bot.bot → Eq n (Nat.card α)
    -/
  · intro h
    /-
      case intro.refine_2
      α : Type u_1
      inst✝² : CommGroup α
      inst✝¹ : IsSimpleGroup α
      inst✝ : Finite α
      h0 : LT.lt 0 (Nat.card α)
      g : α
      hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
      n : Nat
      hn : Dvd.dvd n (Nat.card α)
      h : Eq (Subgroup.zpowers (HPow.hPow g n)) Bot.bot
      ⊢ Eq n (Nat.card α)
    -/
    apply le_antisymm (Nat.le_of_dvd h0 hn)
    /-
      case intro.refine_2
      α : Type u_1
      inst✝² : CommGroup α
      inst✝¹ : IsSimpleGroup α
      inst✝ : Finite α
      h0 : LT.lt 0 (Nat.card α)
      g : α
      hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
      n : Nat
      hn : Dvd.dvd n (Nat.card α)
      h : Eq (Subgroup.zpowers (HPow.hPow g n)) Bot.bot
      ⊢ LE.le (Nat.card α) n
    -/
    rw [← orderOf_eq_card_of_forall_mem_zpowers hg]
    /-
      case intro.refine_2
      α : Type u_1
      inst✝² : CommGroup α
      inst✝¹ : IsSimpleGroup α
      inst✝ : Finite α
      h0 : LT.lt 0 (Nat.card α)
      g : α
      hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
      n : Nat
      hn : Dvd.dvd n (Nat.card α)
      h : Eq (Subgroup.zpowers (HPow.hPow g n)) Bot.bot
      ⊢ LE.le (orderOf g) n
    -/
    apply orderOf_le_of_pow_eq_one (Nat.pos_of_dvd_of_pos hn h0)
    /-
      case intro.refine_2
      α : Type u_1
      inst✝² : CommGroup α
      inst✝¹ : IsSimpleGroup α
      inst✝ : Finite α
      h0 : LT.lt 0 (Nat.card α)
      g : α
      hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
      n : Nat
      hn : Dvd.dvd n (Nat.card α)
      h : Eq (Subgroup.zpowers (HPow.hPow g n)) Bot.bot
      ⊢ Eq (HPow.hPow g n) 1
    -/
    rw [← Subgroup.mem_bot, ← h]
    /-
      case intro.refine_2
      α : Type u_1
      inst✝² : CommGroup α
      inst✝¹ : IsSimpleGroup α
      inst✝ : Finite α
      h0 : LT.lt 0 (Nat.card α)
      g : α
      hg : ∀ (x : α), Membership.mem (Subgroup.zpowers g) x
      n : Nat
      hn : Dvd.dvd n (Nat.card α)
      h : Eq (Subgroup.zpowers (HPow.hPow g n)) Bot.bot
      ⊢ Membership.mem (Subgroup.zpowers (HPow.hPow g n)) (HPow.hPow g n)
    -/
    exact Subgroup.mem_zpowers _
    /-
      🎉 no goals
    -/


@[to_additive]
theorem CommGroup.is_simple_iff_isCyclic_and_prime_card [Finite α] [CommGroup α] :
    IsSimpleGroup α ↔ IsCyclic α ∧ (Nat.card α).Prime := by
  /-
    α : Type u_1
    inst✝¹ : Finite α
    inst✝ : CommGroup α
    ⊢ Iff (IsSimpleGroup α) (And (IsCyclic α) (Nat.Prime (Nat.card α)))
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝¹ : Finite α
      inst✝ : CommGroup α
      ⊢ IsSimpleGroup α → And (IsCyclic α) (Nat.Prime (Nat.card α))
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      inst✝¹ : Finite α
      inst✝ : CommGroup α
      h : IsSimpleGroup α
      ⊢ And (IsCyclic α) (Nat.Prime (Nat.card α))
    -/
    exact ⟨IsSimpleGroup.isCyclic, IsSimpleGroup.prime_card⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝¹ : Finite α
      inst✝ : CommGroup α
      ⊢ And (IsCyclic α) (Nat.Prime (Nat.card α)) → IsSimpleGroup α
    -/
  · rintro ⟨_, hp⟩
    /-
      case mpr.intro
      α : Type u_1
      inst✝¹ : Finite α
      inst✝ : CommGroup α
      left✝ : IsCyclic α
      hp : Nat.Prime (Nat.card α)
      ⊢ IsSimpleGroup α
    -/
    haveI : Fact (Nat.card α).Prime := ⟨hp⟩
    /-
      case mpr.intro
      α : Type u_1
      inst✝¹ : Finite α
      inst✝ : CommGroup α
      left✝ : IsCyclic α
      hp : Nat.Prime (Nat.card α)
      this : Fact (Nat.Prime (Nat.card α))
      ⊢ IsSimpleGroup α
    -/
    exact isSimpleGroup_of_prime_card rfl
    /-
      🎉 no goals
    -/


                                               /-
                                                 α : Type u_1
                                                 G : Type u_2
                                                 G' : Type u_3
                                                 a : α
                                                 n : Int
                                                 ⊢ Eq ((fun x => HSMul.hSMul x 1) n) n
                                               -/
instance : IsAddCyclic ℤ := ⟨1, fun n ↦ ⟨n, by simp only [smul_eq_mul, mul_one]⟩⟩
                                               /-
                                                 🎉 no goals
                                               -/


instance ZMod.instIsAddCyclic (n : ℕ) : IsAddCyclic (ZMod n) :=
  isAddCyclic_of_surjective (Int.castRingHom _) ZMod.intCast_surjective


instance ZMod.instIsSimpleAddGroup {p : ℕ} [Fact p.Prime] : IsSimpleAddGroup (ZMod p) :=
  AddCommGroup.is_simple_iff_isAddCyclic_and_prime_card.2
                       /-
                         α : Type u_1
                         G : Type u_2
                         G' : Type u_3
                         a : α
                         p : Nat
                         inst✝ : Fact (Nat.Prime p)
                         ⊢ Nat.Prime (Nat.card (ZMod p))
                       -/
    ⟨inferInstance, by simpa using (Fact.out : p.Prime)⟩
                       /-
                         🎉 no goals
                       -/


@[to_additive]
theorem IsCyclic.exponent_eq_card [Group α] [IsCyclic α] :
    exponent α = Nat.card α := by
  /-
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : IsCyclic α
    ⊢ Eq (Monoid.exponent α) (Nat.card α)
  -/
  obtain ⟨g, hg⟩ := IsCyclic.exists_ofOrder_eq_natCard (α := α)
  /-
    case intro
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : IsCyclic α
    g : α
    hg : Eq (orderOf g) (Nat.card α)
    ⊢ Eq (Monoid.exponent α) (Nat.card α)
  -/
  apply Nat.dvd_antisymm Group.exponent_dvd_nat_card
  /-
    case intro
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : IsCyclic α
    g : α
    hg : Eq (orderOf g) (Nat.card α)
    ⊢ Dvd.dvd (Nat.card α) (Monoid.exponent α)
  -/
  rw [← hg]
  /-
    case intro
    α : Type u_1
    inst✝¹ : Group α
    inst✝ : IsCyclic α
    g : α
    hg : Eq (orderOf g) (Nat.card α)
    ⊢ Dvd.dvd (orderOf g) (Monoid.exponent α)
  -/
  exact order_dvd_exponent _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem IsCyclic.of_exponent_eq_card [CommGroup α] [Finite α] (h : exponent α = Nat.card α) :
    IsCyclic α :=
  let ⟨_⟩ := nonempty_fintype α
  let ⟨g, _, hg⟩ := Finset.mem_image.mp (Finset.max'_mem _ _)
  isCyclic_of_orderOf_eq_card g <| hg.trans <| exponent_eq_max'_orderOf.symm.trans h


@[to_additive]
theorem IsCyclic.iff_exponent_eq_card [CommGroup α] [Finite α] :
    IsCyclic α ↔ exponent α = Nat.card α :=
  ⟨fun _ => IsCyclic.exponent_eq_card, IsCyclic.of_exponent_eq_card⟩


@[to_additive]
theorem IsCyclic.exponent_eq_zero_of_infinite [Group α] [IsCyclic α] [Infinite α] :
    exponent α = 0 :=
  let ⟨_, hg⟩ := IsCyclic.exists_generator (α := α)
  exponent_eq_zero_of_order_zero <| Infinite.orderOf_eq_zero_of_forall_mem_zpowers hg


@[simp]
protected theorem ZMod.exponent (n : ℕ) : AddMonoid.exponent (ZMod n) = n := by
  /-
    n : Nat
    ⊢ Eq (AddMonoid.exponent (ZMod n)) n
  -/
  rw [IsAddCyclic.exponent_eq_card, Nat.card_zmod]
  /-
    🎉 no goals
  -/


/-- A group of order `p ^ 2` is not cyclic if and only if its exponent is `p`. -/
@[to_additive]
lemma not_isCyclic_iff_exponent_eq_prime [Group α] {p : ℕ} (hp : p.Prime)
    (hα : Nat.card α = p ^ 2) : ¬ IsCyclic α ↔ Monoid.exponent α = p := by
  -- G is a nontrivial fintype of cardinality `p ^ 2`
  /-
    α : Type u_1
    inst✝ : Group α
    p : Nat
    hp : Nat.Prime p
    hα : Eq (Nat.card α) (HPow.hPow p 2)
    ⊢ Iff (Not (IsCyclic α)) (Eq (Monoid.exponent α) p)
  -/
  have : Finite α := Nat.finite_of_card_ne_zero (hα ▸ pow_ne_zero 2 hp.ne_zero)
  have : Nontrivial α := Finite.one_lt_card_iff_nontrivial.mp
    (hα ▸ one_lt_pow₀ hp.one_lt two_ne_zero)
  /- in the forward direction, we apply `exponent_eq_prime_iff`, and the reverse direction follows
  immediately because if `α` has exponent `p`, it has no element of order `p ^ 2`. -/
  refine ⟨fun h_cyc ↦ (Monoid.exponent_eq_prime_iff hp).mpr fun g hg ↦ ?_, fun h_exp h_cyc ↦ by
    obtain (rfl|rfl) := eq_zero_or_one_of_sq_eq_self <| hα ▸ h_exp ▸ (h_cyc.exponent_eq_card).symm
    · exact Nat.not_prime_zero hp
    · exact Nat.not_prime_one hp⟩
  /- we must show every non-identity element has order `p`. By Lagrange's theorem, the only possible
  orders of `g` are `1`, `p`, or `p ^ 2`. It can't be the former because `g ≠ 1`, and it can't
  the latter because the group isn't cyclic. -/
  /-
    α : Type u_1
    inst✝ : Group α
    p : Nat
    hp : Nat.Prime p
    hα : Eq (Nat.card α) (HPow.hPow p 2)
    this✝ : Finite α
    this : Nontrivial α
    h_cyc : Not (IsCyclic α)
    g : α
    hg : Ne g 1
    ⊢ Eq (orderOf g) p
  -/
  have := (Nat.mem_divisors (m := p ^ 2)).mpr ⟨hα ▸ orderOf_dvd_natCard (x := g), by aesop⟩
  simp? [Nat.divisors_prime_pow hp 2] at this says
    simp only [Nat.divisors_prime_pow hp 2, Nat.reduceAdd, Finset.mem_map, Finset.mem_range,
      Function.Embedding.coeFn_mk] at this
  /-
    α : Type u_1
    inst✝ : Group α
    p : Nat
    hp : Nat.Prime p
    hα : Eq (Nat.card α) (HPow.hPow p 2)
    this✝¹ : Finite α
    this✝ : Nontrivial α
    h_cyc : Not (IsCyclic α)
    g : α
    hg : Ne g 1
    this : Exists fun a => And (LT.lt a 3) (Eq (HPow.hPow p a) (orderOf g))
    ⊢ Eq (orderOf g) p
  -/
  obtain ⟨a, ha, ha'⟩ := this
  /-
    case intro.intro
    α : Type u_1
    inst✝ : Group α
    p : Nat
    hp : Nat.Prime p
    hα : Eq (Nat.card α) (HPow.hPow p 2)
    this✝ : Finite α
    this : Nontrivial α
    h_cyc : Not (IsCyclic α)
    g : α
    hg : Ne g 1
    a : Nat
    ha : LT.lt a 3
    ha' : Eq (HPow.hPow p a) (orderOf g)
    ⊢ Eq (orderOf g) p
  -/
  interval_cases a
    /-
      case intro.intro.«0»
      α : Type u_1
      inst✝ : Group α
      p : Nat
      hp : Nat.Prime p
      hα : Eq (Nat.card α) (HPow.hPow p 2)
      this✝ : Finite α
      this : Nontrivial α
      h_cyc : Not (IsCyclic α)
      g : α
      hg : Ne g 1
      a : Nat
      ha : LT.lt 0 3
      ha' : Eq (HPow.hPow p 0) (orderOf g)
      ⊢ Eq (orderOf g) p
    -/
  · exact False.elim <| hg <| orderOf_eq_one_iff.mp <| by aesop
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.«1»
      α : Type u_1
      inst✝ : Group α
      p : Nat
      hp : Nat.Prime p
      hα : Eq (Nat.card α) (HPow.hPow p 2)
      this✝ : Finite α
      this : Nontrivial α
      h_cyc : Not (IsCyclic α)
      g : α
      hg : Ne g 1
      a : Nat
      ha : LT.lt 1 3
      ha' : Eq (HPow.hPow p 1) (orderOf g)
      ⊢ Eq (orderOf g) p
    -/
  · aesop
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.«2»
      α : Type u_1
      inst✝ : Group α
      p : Nat
      hp : Nat.Prime p
      hα : Eq (Nat.card α) (HPow.hPow p 2)
      this✝ : Finite α
      this : Nontrivial α
      h_cyc : Not (IsCyclic α)
      g : α
      hg : Ne g 1
      a : Nat
      ha : LT.lt 2 3
      ha' : Eq (HPow.hPow p 2) (orderOf g)
      ⊢ Eq (orderOf g) p
    -/
  · exact False.elim <| h_cyc <| isCyclic_of_orderOf_eq_card g <| by aesop
    /-
      🎉 no goals
    -/


/-- The kernel of `zmultiplesHom G g` is equal to the additive subgroup generated by
    `addOrderOf g`. -/
theorem zmultiplesHom_ker_eq [AddGroup G] (g : G) :
    (zmultiplesHom G g).ker = zmultiples ↑(addOrderOf g) := by
  /-
    G : Type u_2
    inst✝ : AddGroup G
    g : G
    ⊢ Eq ((zmultiplesHom G) g).ker (AddSubgroup.zmultiples ↑(addOrderOf g))
  -/
  ext
  simp_rw [AddMonoidHom.mem_ker, mem_zmultiples_iff, zmultiplesHom_apply,
    ← addOrderOf_dvd_iff_zsmul_eq_zero, zsmul_eq_mul', Int.cast_id, dvd_def, eq_comm]


/-- The kernel of `zpowersHom G g` is equal to the subgroup generated by `orderOf g`. -/
theorem zpowersHom_ker_eq [Group G] (g : G) :
    (zpowersHom G g).ker = zpowers (Multiplicative.ofAdd ↑(orderOf g)) :=
  congr_arg AddSubgroup.toSubgroup <| zmultiplesHom_ker_eq (Additive.ofMul g)


/-- The isomorphism from `ZMod n` to any cyclic additive group of `Nat.card` equal to `n`. -/
noncomputable def zmodAddCyclicAddEquiv [AddGroup G] (h : IsAddCyclic G) :
    ZMod (Nat.card G) ≃+ G := by
  /-
    α : Type u_1
    G : Type u_2
    G' : Type u_3
    a : α
    inst✝ : AddGroup G
    h : IsAddCyclic G
    ⊢ AddEquiv (ZMod (Nat.card G)) G
  -/
  let n := Nat.card G
  /-
    α : Type u_1
    G : Type u_2
    G' : Type u_3
    a : α
    inst✝ : AddGroup G
    h : IsAddCyclic G
    n : Nat := Nat.card G
    ⊢ AddEquiv (ZMod (Nat.card G)) G
  -/
  let ⟨g, surj⟩ := Classical.indefiniteDescription _ h.exists_generator
  have kereq : ((zmultiplesHom G) g).ker = zmultiples ↑(Nat.card G) := by
    rw [zmultiplesHom_ker_eq]
    congr
    rw [← Nat.card_zmultiples]
    exact Nat.card_congr (Equiv.subtypeUnivEquiv surj)
  exact Int.quotientZMultiplesNatEquivZMod n
    |>.symm.trans <| QuotientAddGroup.quotientAddEquivOfEq kereq
    |>.symm.trans <| QuotientAddGroup.quotientKerEquivOfSurjective (zmultiplesHom G g) surj


/-- The isomorphism from `Multiplicative (ZMod n)` to any cyclic group of `Nat.card` equal to `n`.
-/
noncomputable def zmodCyclicMulEquiv [Group G] (h : IsCyclic G) :
    Multiplicative (ZMod (Nat.card G)) ≃* G :=
  AddEquiv.toMultiplicative <| zmodAddCyclicAddEquiv <| isAddCyclic_additive_iff.2 h


/-- Two cyclic additive groups of the same cardinality are isomorphic. -/
noncomputable def addEquivOfAddCyclicCardEq [AddGroup G] [AddGroup G'] [hG : IsAddCyclic G]
    [hH : IsAddCyclic G'] (hcard : Nat.card G = Nat.card G') : G ≃+ G' := hcard ▸
  zmodAddCyclicAddEquiv hG |>.symm.trans (zmodAddCyclicAddEquiv hH)


/-- Two cyclic groups of the same cardinality are isomorphic. -/
@[to_additive existing]
noncomputable def mulEquivOfCyclicCardEq [Group G] [Group G'] [hG : IsCyclic G]
    [hH : IsCyclic G'] (hcard : Nat.card G = Nat.card G') : G ≃* G' := hcard ▸
  zmodCyclicMulEquiv hG |>.symm.trans (zmodCyclicMulEquiv hH)


/-- Two groups of the same prime cardinality are isomorphic. -/
@[to_additive "Two additive groups of the same prime cardinality are isomorphic."]
noncomputable def mulEquivOfPrimeCardEq {p : ℕ} [Group G] [Group G']
    [Fact p.Prime] (hG : Nat.card G = p) (hH : Nat.card G' = p) : G ≃* G' := by
  /-
    α : Type u_1
    G : Type u_2
    G' : Type u_3
    a : α
    p : Nat
    inst✝² : Group G
    inst✝¹ : Group G'
    inst✝ : Fact (Nat.Prime p)
    hG : Eq (Nat.card G) p
    hH : Eq (Nat.card G') p
    ⊢ MulEquiv G G'
  -/
  have hGcyc := isCyclic_of_prime_card hG
  /-
    α : Type u_1
    G : Type u_2
    G' : Type u_3
    a : α
    p : Nat
    inst✝² : Group G
    inst✝¹ : Group G'
    inst✝ : Fact (Nat.Prime p)
    hG : Eq (Nat.card G) p
    hH : Eq (Nat.card G') p
    hGcyc : IsCyclic G
    ⊢ MulEquiv G G'
  -/
  have hHcyc := isCyclic_of_prime_card hH
  /-
    α : Type u_1
    G : Type u_2
    G' : Type u_3
    a : α
    p : Nat
    inst✝² : Group G
    inst✝¹ : Group G'
    inst✝ : Fact (Nat.Prime p)
    hG : Eq (Nat.card G) p
    hH : Eq (Nat.card G') p
    hGcyc : IsCyclic G
    hHcyc : IsCyclic G'
    ⊢ MulEquiv G G'
  -/
  apply mulEquivOfCyclicCardEq
  /-
    case hcard
    α : Type u_1
    G : Type u_2
    G' : Type u_3
    a : α
    p : Nat
    inst✝² : Group G
    inst✝¹ : Group G'
    inst✝ : Fact (Nat.Prime p)
    hG : Eq (Nat.card G) p
    hH : Eq (Nat.card G') p
    hGcyc : IsCyclic G
    hHcyc : IsCyclic G'
    ⊢ Eq (Nat.card G) (Nat.card G')
  -/
  exact hG.trans hH.symm
  /-
    🎉 no goals
  -/


variable (G) in
/-- The automorphism group of a cyclic group is isomorphic to the multiplicative group of ZMod. -/
@[simps!]
noncomputable def IsCyclic.mulAutMulEquiv [Group G] [h : IsCyclic G] :
    MulAut G ≃* (ZMod (Nat.card G))ˣ :=
  ((MulAut.congr (zmodCyclicMulEquiv h)).symm.trans
    (MulAutMultiplicative (ZMod (Nat.card G)))).trans (ZMod.AddAutEquivUnits (Nat.card G))


variable (G) in
theorem IsCyclic.card_mulAut [Group G] [Finite G] [h : IsCyclic G] :
    Nat.card (MulAut G) = Nat.totient (Nat.card G) := by
  /-
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : Finite G
    h : IsCyclic G
    ⊢ Eq (Nat.card (MulAut G)) (Nat.card G).totient
  -/
  have : NeZero (Nat.card G) := ⟨Nat.card_pos.ne'⟩
  /-
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : Finite G
    h : IsCyclic G
    this : NeZero (Nat.card G)
    ⊢ Eq (Nat.card (MulAut G)) (Nat.card G).totient
  -/
  rw [← ZMod.card_units_eq_totient, ← Nat.card_eq_fintype_card]
  /-
    G : Type u_2
    inst✝¹ : Group G
    inst✝ : Finite G
    h : IsCyclic G
    this : NeZero (Nat.card G)
    ⊢ Eq (Nat.card (MulAut G)) (Nat.card (Units (ZMod (Nat.card G))))
  -/
  exact Nat.card_congr (mulAutMulEquiv G)
  /-
    🎉 no goals
  -/


/-- If `g` generates the group `G` and `g'` is an element of another group `G'` whose order
divides that of `g`, then there is a homomorphism `G →* G'` mapping `g` to `g'`. -/
@[to_additive
   "If `g` generates the additive group `G` and `g'` is an element of another additive group `G'`
   whose order divides that of `g`, then there is a homomorphism `G →+ G'` mapping `g` to `g'`."]
noncomputable
def monoidHomOfForallMemZpowers : G →* G' where
  toFun x := g' ^ (Classical.choose <| mem_zpowers_iff.mp <| hg x)
  map_one' := orderOf_dvd_iff_zpow_eq_one.mp <|
                (Int.natCast_dvd_natCast.mpr hg').trans <| orderOf_dvd_iff_zpow_eq_one.mpr <|
                Classical.choose_spec <| mem_zpowers_iff.mp <| hg 1
  map_mul' x y := by
    /-
      α : Type u_1
      G : Type u_2
      G' : Type u_3
      a : α
      inst✝¹ : Group G
      inst✝ : Group G'
      g : G
      hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
      g' : G'
      hg' : Dvd.dvd (orderOf g') (orderOf g)
      x y : G
      ⊢ Eq ({ toFun := fun x => HPow.hPow g' (Classical.choose ⋯), map_one' := ⋯ }.t …
    -/
    simp only [← zpow_add, zpow_eq_zpow_iff_modEq]
    /-
      α : Type u_1
      G : Type u_2
      G' : Type u_3
      a : α
      inst✝¹ : Group G
      inst✝ : Group G'
      g : G
      hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
      g' : G'
      hg' : Dvd.dvd (orderOf g') (orderOf g)
      x y : G
      ⊢ (↑(orderOf g')).ModEq (Classical.choose ⋯) (HAdd.hAdd (Classical.choose ⋯) ( …
    -/
    apply Int.ModEq.of_dvd (Int.natCast_dvd_natCast.mpr hg')
    /-
      α : Type u_1
      G : Type u_2
      G' : Type u_3
      a : α
      inst✝¹ : Group G
      inst✝ : Group G'
      g : G
      hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
      g' : G'
      hg' : Dvd.dvd (orderOf g') (orderOf g)
      x y : G
      ⊢ (↑(orderOf g)).ModEq (Classical.choose ⋯) (HAdd.hAdd (Classical.choose ⋯) (C …
    -/
    rw [← zpow_eq_zpow_iff_modEq, zpow_add]
    /-
      α : Type u_1
      G : Type u_2
      G' : Type u_3
      a : α
      inst✝¹ : Group G
      inst✝ : Group G'
      g : G
      hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
      g' : G'
      hg' : Dvd.dvd (orderOf g') (orderOf g)
      x y : G
      ⊢ Eq (HPow.hPow g (Classical.choose ⋯)) (HMul.hMul (HPow.hPow g (Classical.cho …
    -/
    simp only [fun x ↦ Classical.choose_spec <| mem_zpowers_iff.mp <| hg x]
    /-
      🎉 no goals
    -/


@[to_additive (attr := simp)]
lemma monoidHomOfForallMemZpowers_apply_gen :
    monoidHomOfForallMemZpowers hg hg' g = g' := by
  /-
    G : Type u_2
    G' : Type u_3
    inst✝¹ : Group G
    inst✝ : Group G'
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    g' : G'
    hg' : Dvd.dvd (orderOf g') (orderOf g)
    ⊢ Eq ((monoidHomOfForallMemZpowers hg hg') g) g'
  -/
  simp only [monoidHomOfForallMemZpowers, MonoidHom.coe_mk, OneHom.coe_mk]
  /-
    G : Type u_2
    G' : Type u_3
    inst✝¹ : Group G
    inst✝ : Group G'
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    g' : G'
    hg' : Dvd.dvd (orderOf g') (orderOf g)
    ⊢ Eq (HPow.hPow g' (Classical.choose ⋯)) g'
  -/
  nth_rw 2 [← zpow_one g']
  /-
    G : Type u_2
    G' : Type u_3
    inst✝¹ : Group G
    inst✝ : Group G'
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    g' : G'
    hg' : Dvd.dvd (orderOf g') (orderOf g)
    ⊢ Eq (HPow.hPow g' (Classical.choose ⋯)) (HPow.hPow g' 1)
  -/
  rw [zpow_eq_zpow_iff_modEq]
  /-
    G : Type u_2
    G' : Type u_3
    inst✝¹ : Group G
    inst✝ : Group G'
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    g' : G'
    hg' : Dvd.dvd (orderOf g') (orderOf g)
    ⊢ (↑(orderOf g')).ModEq (Classical.choose ⋯) 1
  -/
  apply Int.ModEq.of_dvd (Int.natCast_dvd_natCast.mpr hg')
  /-
    G : Type u_2
    G' : Type u_3
    inst✝¹ : Group G
    inst✝ : Group G'
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    g' : G'
    hg' : Dvd.dvd (orderOf g') (orderOf g)
    ⊢ (↑(orderOf g)).ModEq (Classical.choose ⋯) 1
  -/
  rw [← zpow_eq_zpow_iff_modEq, zpow_one]
  /-
    G : Type u_2
    G' : Type u_3
    inst✝¹ : Group G
    inst✝ : Group G'
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    g' : G'
    hg' : Dvd.dvd (orderOf g') (orderOf g)
    ⊢ Eq (HPow.hPow g (Classical.choose ⋯)) g
  -/
  exact Classical.choose_spec <| mem_zpowers_iff.mp <| hg g
  /-
    🎉 no goals
  -/


/-- Two group homomorphisms `G →* G'` are equal if and only if they agree on a generator of `G`. -/
@[to_additive
   "Two homomorphisms `G →+ G'` of additive groups are equal if and only if they agree
   on a generator of `G`."]
lemma MonoidHom.eq_iff_eq_on_generator (f₁ f₂ : G →* G') : f₁ = f₂ ↔ f₁ g = f₂ g := by
  /-
    G : Type u_2
    G' : Type u_3
    inst✝¹ : Group G
    inst✝ : Group G'
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    f₁ f₂ : MonoidHom G G'
    ⊢ Iff (Eq f₁ f₂) (Eq (f₁ g) (f₂ g))
  -/
  rw [DFunLike.ext_iff]
  /-
    G : Type u_2
    G' : Type u_3
    inst✝¹ : Group G
    inst✝ : Group G'
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    f₁ f₂ : MonoidHom G G'
    ⊢ Iff (∀ (x : G), Eq (f₁ x) (f₂ x)) (Eq (f₁ g) (f₂ g))
  -/
  refine ⟨fun H ↦ H g, fun H x ↦ ?_⟩
  /-
    G : Type u_2
    G' : Type u_3
    inst✝¹ : Group G
    inst✝ : Group G'
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    f₁ f₂ : MonoidHom G G'
    H : Eq (f₁ g) (f₂ g)
    x : G
    ⊢ Eq (f₁ x) (f₂ x)
  -/
  obtain ⟨n, hn⟩ := mem_zpowers_iff.mp <| hg x
  /-
    case intro
    G : Type u_2
    G' : Type u_3
    inst✝¹ : Group G
    inst✝ : Group G'
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    f₁ f₂ : MonoidHom G G'
    H : Eq (f₁ g) (f₂ g)
    x : G
    n : Int
    hn : Eq (HPow.hPow g n) x
    ⊢ Eq (f₁ x) (f₂ x)
  -/
  rw [← hn, map_zpow, map_zpow, H]
  /-
    🎉 no goals
  -/


/-- Two group isomorphisms `G ≃* G'` are equal if and only if they agree on a generator of `G`. -/
@[to_additive
   "Two isomorphisms `G ≃+ G'` of additive groups are equal if and only if they agree
   on a generator of `G`."]
lemma MulEquiv.eq_iff_eq_on_generator (f₁ f₂ : G ≃* G') : f₁ = f₂ ↔ f₁ g = f₂ g :=
  (Function.Injective.eq_iff toMonoidHom_injective).symm.trans <|
    MonoidHom.eq_iff_eq_on_generator hg ..


/-- Given two groups that are generated by elements `g` and `g'` of the same order,
we obtain an isomorphism sending `g` to `g'`. -/
@[to_additive
   "Given two additive groups that are generated by elements `g` and `g'` of the same order,
   we obtain an isomorphism sending `g` to `g'`."]
noncomputable
def mulEquivOfOrderOfEq : G ≃* G' := by
  refine MonoidHom.toMulEquiv (monoidHomOfForallMemZpowers hg h.symm.dvd)
    (monoidHomOfForallMemZpowers hg' h.dvd) ?_ ?_ <;>
  /-
    case refine_1
    α : Type u_1
    G : Type u_2
    G' : Type u_3
    a : α
    inst✝¹ : Group G
    inst✝ : Group G'
    g : G
    hg : ∀ (x : G), Membership.mem (Subgroup.zpowers g) x
    g' : G'
    hg' : ∀ (x : G'), Membership.mem (Subgroup.zpowers g') x
    h : Eq (orderOf g) (orderOf g')
    ⊢ Eq ((monoidHomOfForallMemZpowers hg' ⋯).comp (monoidHomOfForallMemZpowers hg …
  -/
  refine (MonoidHom.eq_iff_eq_on_generator (by assumption) _ _).mpr ?_ <;>
  simp only [MonoidHom.coe_comp, Function.comp_apply, monoidHomOfForallMemZpowers_apply_gen,
    MonoidHom.id_apply]


@[to_additive (attr := simp)]
lemma mulEquivOfOrderOfEq_apply_gen : mulEquivOfOrderOfEq hg hg' h g = g' :=
  monoidHomOfForallMemZpowers_apply_gen hg h.symm.dvd


@[to_additive (attr := simp)]
lemma mulEquivOfOrderOfEq_symm :
    (mulEquivOfOrderOfEq hg hg' h).symm = mulEquivOfOrderOfEq hg' hg h.symm := rfl


@[to_additive] -- `simp` can prove this by a combination of the two preceding lemmas
lemma mulEquivOfOrderOfEq_symm_apply_gen : (mulEquivOfOrderOfEq hg hg' h).symm g' = g :=
  monoidHomOfForallMemZpowers_apply_gen hg' h.dvd



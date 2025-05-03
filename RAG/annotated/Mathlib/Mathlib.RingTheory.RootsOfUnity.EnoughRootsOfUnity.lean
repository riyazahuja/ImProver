/-- This is a type class recording that a commutative monoid `M` contains primitive `n`th
roots of unity and such that the group of `n`th roots of unity is cyclic.

Such monoids are suitable targets in the context of duality statements for groups
of exponent `n`. -/
class HasEnoughRootsOfUnity (M : Type*) [CommMonoid M] (n : ℕ) where
  prim : ∃ m : M, IsPrimitiveRoot m n
  cyc : IsCyclic <| rootsOfUnity n M


lemma exists_primitiveRoot (M : Type*) [CommMonoid M] (n : ℕ) [HasEnoughRootsOfUnity M n] :
    ∃ ζ : M, IsPrimitiveRoot ζ n :=
  HasEnoughRootsOfUnity.prim


instance rootsOfUnity_isCyclic (M : Type*) [CommMonoid M] (n : ℕ) [HasEnoughRootsOfUnity M n] :
    IsCyclic (rootsOfUnity n M) :=
  HasEnoughRootsOfUnity.cyc


/-- If `HasEnoughRootsOfUnity M n` and `m ∣ n`, then also `HasEnoughRootsOfUnity M m`. -/
lemma of_dvd (M : Type*) [CommMonoid M] {m n : ℕ} [NeZero n] (hmn : m ∣ n)
    [HasEnoughRootsOfUnity M n] :
    HasEnoughRootsOfUnity M m where
  prim :=
    have ⟨ζ, hζ⟩ := exists_primitiveRoot M n
    have ⟨k, hk⟩ := hmn
    ⟨ζ ^ k, IsPrimitiveRoot.pow (NeZero.pos n) hζ (mul_comm m k ▸ hk)⟩
  cyc := Subgroup.isCyclic_of_le <| rootsOfUnity_le_of_dvd hmn


/-- If `M` satisfies `HasEnoughRootsOfUnity`, then the group of `n`th roots of unity
in `M` is finite. -/
instance finite_rootsOfUnity (M : Type*) [CommMonoid M] (n : ℕ) [NeZero n]
    [HasEnoughRootsOfUnity M n] :
    Finite <| rootsOfUnity n M := by
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : HasEnoughRootsOfUnity M n
    ⊢ Finite (Subtype fun x => Membership.mem (rootsOfUnity n M) x)
  -/
  have := rootsOfUnity_isCyclic M n
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : HasEnoughRootsOfUnity M n
    this : IsCyclic (Subtype fun x => Membership.mem (rootsOfUnity n M) x)
    ⊢ Finite (Subtype fun x => Membership.mem (rootsOfUnity n M) x)
  -/
  obtain ⟨g, hg⟩ := IsCyclic.exists_generator (α := rootsOfUnity n M)
  /-
    case intro
    M : Type u_1
    inst✝² : CommMonoid M
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : HasEnoughRootsOfUnity M n
    this : IsCyclic (Subtype fun x => Membership.mem (rootsOfUnity n M) x)
    g : Subtype fun x => Membership.mem (rootsOfUnity n M) x
    hg : ∀ (x : Subtype fun x => Membership.mem (rootsOfUnity n M) x), Membership. …
    ⊢ Finite (Subtype fun x => Membership.mem (rootsOfUnity n M) x)
  -/
  have hg' : g ^ n = 1 := OneMemClass.coe_eq_one.mp g.prop
  /-
    case intro
    M : Type u_1
    inst✝² : CommMonoid M
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : HasEnoughRootsOfUnity M n
    this : IsCyclic (Subtype fun x => Membership.mem (rootsOfUnity n M) x)
    g : Subtype fun x => Membership.mem (rootsOfUnity n M) x
    hg : ∀ (x : Subtype fun x => Membership.mem (rootsOfUnity n M) x), Membership. …
    hg' : Eq (HPow.hPow g n) 1
    ⊢ Finite (Subtype fun x => Membership.mem (rootsOfUnity n M) x)
  -/
  let f (j : ZMod n) : rootsOfUnity n M := g ^ (j.val : ℤ)
  /-
    case intro
    M : Type u_1
    inst✝² : CommMonoid M
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : HasEnoughRootsOfUnity M n
    this : IsCyclic (Subtype fun x => Membership.mem (rootsOfUnity n M) x)
    g : Subtype fun x => Membership.mem (rootsOfUnity n M) x
    hg : ∀ (x : Subtype fun x => Membership.mem (rootsOfUnity n M) x), Membership. …
    hg' : Eq (HPow.hPow g n) 1
    f : ZMod n → Subtype fun x => Membership.mem (rootsOfUnity n M) x := fun j =>  …
    ⊢ Finite (Subtype fun x => Membership.mem (rootsOfUnity n M) x)
  -/
  refine Finite.of_surjective f fun x ↦ ?_
  /-
    case intro
    M : Type u_1
    inst✝² : CommMonoid M
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : HasEnoughRootsOfUnity M n
    this : IsCyclic (Subtype fun x => Membership.mem (rootsOfUnity n M) x)
    g : Subtype fun x => Membership.mem (rootsOfUnity n M) x
    hg : ∀ (x : Subtype fun x => Membership.mem (rootsOfUnity n M) x), Membership. …
    hg' : Eq (HPow.hPow g n) 1
    f : ZMod n → Subtype fun x => Membership.mem (rootsOfUnity n M) x := fun j =>  …
    x : Subtype fun x => Membership.mem (rootsOfUnity n M) x
    ⊢ Exists fun a => Eq (f a) x
  -/
  obtain ⟨k, hk⟩ := Subgroup.mem_zpowers_iff.mp <| hg x
  /-
    case intro.intro
    M : Type u_1
    inst✝² : CommMonoid M
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : HasEnoughRootsOfUnity M n
    this : IsCyclic (Subtype fun x => Membership.mem (rootsOfUnity n M) x)
    g : Subtype fun x => Membership.mem (rootsOfUnity n M) x
    hg : ∀ (x : Subtype fun x => Membership.mem (rootsOfUnity n M) x), Membership. …
    hg' : Eq (HPow.hPow g n) 1
    f : ZMod n → Subtype fun x => Membership.mem (rootsOfUnity n M) x := fun j =>  …
    x : Subtype fun x => Membership.mem (rootsOfUnity n M) x
    k : Int
    hk : Eq (HPow.hPow g k) x
    ⊢ Exists fun a => Eq (f a) x
  -/
  refine ⟨k, ?_⟩
  /-
    case intro.intro
    M : Type u_1
    inst✝² : CommMonoid M
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : HasEnoughRootsOfUnity M n
    this : IsCyclic (Subtype fun x => Membership.mem (rootsOfUnity n M) x)
    g : Subtype fun x => Membership.mem (rootsOfUnity n M) x
    hg : ∀ (x : Subtype fun x => Membership.mem (rootsOfUnity n M) x), Membership. …
    hg' : Eq (HPow.hPow g n) 1
    f : ZMod n → Subtype fun x => Membership.mem (rootsOfUnity n M) x := fun j =>  …
    x : Subtype fun x => Membership.mem (rootsOfUnity n M) x
    k : Int
    hk : Eq (HPow.hPow g k) x
    ⊢ Eq (f ↑k) x
  -/
  simpa only [ZMod.natCast_val, ← hk, f, ZMod.coe_intCast] using (zpow_eq_zpow_emod' k hg').symm
  /-
    🎉 no goals
  -/


/-- If `M` satisfies `HasEnoughRootsOfUnity`, then the group of `n`th roots of unity
in `M` (is cyclic and) has order `n`. -/
lemma natCard_rootsOfUnity (M : Type*) [CommMonoid M] (n : ℕ) [NeZero n]
    [HasEnoughRootsOfUnity M n] :
    Nat.card (rootsOfUnity n M) = n := by
  /-
    M : Type u_1
    inst✝² : CommMonoid M
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : HasEnoughRootsOfUnity M n
    ⊢ Eq (Nat.card (Subtype fun x => Membership.mem (rootsOfUnity n M) x)) n
  -/
  obtain ⟨ζ, h⟩ := exists_primitiveRoot M n
  /-
    case intro
    M : Type u_1
    inst✝² : CommMonoid M
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : HasEnoughRootsOfUnity M n
    ζ : M
    h : IsPrimitiveRoot ζ n
    ⊢ Eq (Nat.card (Subtype fun x => Membership.mem (rootsOfUnity n M) x)) n
  -/
  rw [← IsCyclic.exponent_eq_card]
  /-
    case intro
    M : Type u_1
    inst✝² : CommMonoid M
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : HasEnoughRootsOfUnity M n
    ζ : M
    h : IsPrimitiveRoot ζ n
    ⊢ Eq (Monoid.exponent (Subtype fun x => Membership.mem (rootsOfUnity n M) x)) n
  -/
  refine dvd_antisymm ?_ ?_
    /-
      case intro.refine_1
      M : Type u_1
      inst✝² : CommMonoid M
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : HasEnoughRootsOfUnity M n
      ζ : M
      h : IsPrimitiveRoot ζ n
      ⊢ Dvd.dvd (Monoid.exponent (Subtype fun x => Membership.mem (rootsOfUnity n M) …
    -/
  · exact Monoid.exponent_dvd_of_forall_pow_eq_one fun g ↦ OneMemClass.coe_eq_one.mp g.prop
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      M : Type u_1
      inst✝² : CommMonoid M
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : HasEnoughRootsOfUnity M n
      ζ : M
      h : IsPrimitiveRoot ζ n
      ⊢ Dvd.dvd n (Monoid.exponent (Subtype fun x => Membership.mem (rootsOfUnity n  …
    -/
  · nth_rewrite 1 [h.eq_orderOf]
    /-
      case intro.refine_2
      M : Type u_1
      inst✝² : CommMonoid M
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : HasEnoughRootsOfUnity M n
      ζ : M
      h : IsPrimitiveRoot ζ n
      ⊢ Dvd.dvd (orderOf ζ) (Monoid.exponent (Subtype fun x => Membership.mem (roots …
    -/
    rw [← (h.isUnit <| NeZero.pos n).unit_spec, orderOf_units]
    /-
      case intro.refine_2
      M : Type u_1
      inst✝² : CommMonoid M
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : HasEnoughRootsOfUnity M n
      ζ : M
      h : IsPrimitiveRoot ζ n
      ⊢ Dvd.dvd (orderOf ⋯.unit) (Monoid.exponent (Subtype fun x => Membership.mem ( …
    -/
    let ζ' : rootsOfUnity n M := ⟨(h.isUnit <| NeZero.pos n).unit, ?_⟩
      /-
        case intro.refine_2.refine_2
        M : Type u_1
        inst✝² : CommMonoid M
        n : Nat
        inst✝¹ : NeZero n
        inst✝ : HasEnoughRootsOfUnity M n
        ζ : M
        h : IsPrimitiveRoot ζ n
        ζ' : Subtype fun x => Membership.mem (rootsOfUnity n M) x := ⟨⋯.unit, ?intro.r …
        ⊢ Dvd.dvd (orderOf ⋯.unit) (Monoid.exponent (Subtype fun x => Membership.mem ( …
      -/
    · rw [← Subgroup.orderOf_mk]
      /-
        case intro.refine_2.refine_2
        M : Type u_1
        inst✝² : CommMonoid M
        n : Nat
        inst✝¹ : NeZero n
        inst✝ : HasEnoughRootsOfUnity M n
        ζ : M
        h : IsPrimitiveRoot ζ n
        ζ' : Subtype fun x => Membership.mem (rootsOfUnity n M) x := ⟨⋯.unit, ?intro.r …
        ⊢ Dvd.dvd (orderOf ⟨⋯.unit, ?intro.refine_2.refine_2.ha⟩) (Monoid.exponent (Su …
      -/
      exact Monoid.order_dvd_exponent ζ'
      /-
        🎉 no goals
      -/
    /-
      case intro.refine_2.refine_1
      M : Type u_1
      inst✝² : CommMonoid M
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : HasEnoughRootsOfUnity M n
      ζ : M
      h : IsPrimitiveRoot ζ n
      ⊢ Membership.mem (rootsOfUnity n M) ⋯.unit
    -/
    simp only [mem_rootsOfUnity, PNat.mk_coe]
    /-
      case intro.refine_2.refine_1
      M : Type u_1
      inst✝² : CommMonoid M
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : HasEnoughRootsOfUnity M n
      ζ : M
      h : IsPrimitiveRoot ζ n
      ⊢ Eq (HPow.hPow ⋯.unit n) 1
    -/
    rw [← Units.eq_iff, Units.val_pow_eq_pow_val, IsUnit.unit_spec, h.pow_eq_one, Units.val_one]
    /-
      🎉 no goals
    -/


/-- The group of group homomorphims from a finite cyclic group `G` of order `n` into the
group of units of a ring `M` with all roots of unity is isomorphic to `G` -/
lemma IsCyclic.monoidHom_equiv_self (G M : Type*) [CommGroup G] [Finite G]
    [IsCyclic G] [CommMonoid M] [HasEnoughRootsOfUnity M (Nat.card G)] :
    Nonempty ((G →* Mˣ) ≃* G) := by
  /-
    G : Type u_1
    M : Type u_2
    inst✝⁴ : CommGroup G
    inst✝³ : Finite G
    inst✝² : IsCyclic G
    inst✝¹ : CommMonoid M
    inst✝ : HasEnoughRootsOfUnity M (Nat.card G)
    ⊢ Nonempty (MulEquiv (MonoidHom G (Units M)) G)
  -/
  have : NeZero (Nat.card G) := ⟨Nat.card_pos.ne'⟩
  /-
    G : Type u_1
    M : Type u_2
    inst✝⁴ : CommGroup G
    inst✝³ : Finite G
    inst✝² : IsCyclic G
    inst✝¹ : CommMonoid M
    inst✝ : HasEnoughRootsOfUnity M (Nat.card G)
    this : NeZero (Nat.card G)
    ⊢ Nonempty (MulEquiv (MonoidHom G (Units M)) G)
  -/
  have hord := HasEnoughRootsOfUnity.natCard_rootsOfUnity M (Nat.card G)
  /-
    G : Type u_1
    M : Type u_2
    inst✝⁴ : CommGroup G
    inst✝³ : Finite G
    inst✝² : IsCyclic G
    inst✝¹ : CommMonoid M
    inst✝ : HasEnoughRootsOfUnity M (Nat.card G)
    this : NeZero (Nat.card G)
    hord : Eq (Nat.card (Subtype fun x => Membership.mem (rootsOfUnity (Nat.card G …
    ⊢ Nonempty (MulEquiv (MonoidHom G (Units M)) G)
  -/
  let e := (IsCyclic.monoidHom_mulEquiv_rootsOfUnity G Mˣ).some
  /-
    G : Type u_1
    M : Type u_2
    inst✝⁴ : CommGroup G
    inst✝³ : Finite G
    inst✝² : IsCyclic G
    inst✝¹ : CommMonoid M
    inst✝ : HasEnoughRootsOfUnity M (Nat.card G)
    this : NeZero (Nat.card G)
    hord : Eq (Nat.card (Subtype fun x => Membership.mem (rootsOfUnity (Nat.card G …
    e : MulEquiv (MonoidHom G (Units M)) (Subtype fun x => Membership.mem (rootsOf …
    ⊢ Nonempty (MulEquiv (MonoidHom G (Units M)) G)
  -/
  exact ⟨e.trans (rootsOfUnityUnitsMulEquiv M (Nat.card G)) |>.trans (mulEquivOfCyclicCardEq hord)⟩
  /-
    🎉 no goals
  -/



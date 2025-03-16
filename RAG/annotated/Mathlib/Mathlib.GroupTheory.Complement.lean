/-- `S` and `T` are complements if `(*) : S × T → G` is a bijection.
  This notion generalizes left transversals, right transversals, and complementary subgroups. -/
@[to_additive "`S` and `T` are complements if `(+) : S × T → G` is a bijection"]
def IsComplement : Prop :=
  Function.Bijective fun x : S × T => x.1.1 * x.2.1


/-- `H` and `K` are complements if `(*) : H × K → G` is a bijection -/
@[to_additive "`H` and `K` are complements if `(+) : H × K → G` is a bijection"]
abbrev IsComplement' :=
  IsComplement (H : Set G) (K : Set G)


/-- The set of left-complements of `T : Set G` -/
@[to_additive (attr := deprecated IsComplement (since := "2024-12-18"))
"The set of left-complements of `T : Set G`"]
def leftTransversals : Set (Set G) :=
  { S : Set G | IsComplement S T }


/-- The set of right-complements of `S : Set G` -/
@[to_additive (attr := deprecated IsComplement (since := "2024-12-18"))
"The set of right-complements of `S : Set G`"]
def rightTransversals : Set (Set G) :=
  { T : Set G | IsComplement S T }


@[to_additive]
theorem isComplement'_def : IsComplement' H K ↔ IsComplement (H : Set G) (K : Set G) :=
  Iff.rfl


@[to_additive]
theorem isComplement_iff_existsUnique :
    IsComplement S T ↔ ∀ g : G, ∃! x : S × T, x.1.1 * x.2.1 = g :=
  Function.bijective_iff_existsUnique _


@[to_additive]
theorem IsComplement.existsUnique (h : IsComplement S T) (g : G) :
    ∃! x : S × T, x.1.1 * x.2.1 = g :=
  isComplement_iff_existsUnique.mp h g


@[to_additive]
theorem IsComplement'.symm (h : IsComplement' H K) : IsComplement' K H := by
  let ϕ : H × K ≃ K × H :=
    Equiv.mk (fun x => ⟨x.2⁻¹, x.1⁻¹⟩) (fun x => ⟨x.2⁻¹, x.1⁻¹⟩)
      (fun x => Prod.ext (inv_inv _) (inv_inv _)) fun x => Prod.ext (inv_inv _) (inv_inv _)
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    h : H.IsComplement' K
    ϕ : Equiv (Prod (Subtype fun x => Membership.mem H x) (Subtype fun x => Member …
    ⊢ K.IsComplement' H
  -/
  let ψ : G ≃ G := Equiv.mk (fun g : G => g⁻¹) (fun g : G => g⁻¹) inv_inv inv_inv
  suffices hf : (ψ ∘ fun x : H × K => x.1.1 * x.2.1) = (fun x : K × H => x.1.1 * x.2.1) ∘ ϕ by
    rw [isComplement'_def, IsComplement, ← Equiv.bijective_comp ϕ]
    apply (congr_arg Function.Bijective hf).mp -- Porting note: This was a `rw` in mathlib3
    rwa [ψ.comp_bijective]
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    h : H.IsComplement' K
    ϕ : Equiv (Prod (Subtype fun x => Membership.mem H x) (Subtype fun x => Member …
    ψ : Equiv G G := { toFun := fun g => Inv.inv g, invFun := fun g => Inv.inv g,  …
    ⊢ Eq (Function.comp ⇑ψ fun x => HMul.hMul ↑x.1 ↑x.2) (Function.comp (fun x =>  …
  -/
  exact funext fun x => mul_inv_rev _ _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem isComplement'_comm : IsComplement' H K ↔ IsComplement' K H :=
  ⟨IsComplement'.symm, IsComplement'.symm⟩


@[to_additive]
theorem isComplement_univ_singleton {g : G} : IsComplement (univ : Set G) {g} :=
  ⟨fun ⟨_, _, rfl⟩ ⟨_, _, rfl⟩ h => Prod.ext (Subtype.ext (mul_right_cancel h)) rfl, fun x =>
    ⟨⟨⟨x * g⁻¹, ⟨⟩⟩, g, rfl⟩, inv_mul_cancel_right x g⟩⟩


@[to_additive]
theorem isComplement_singleton_univ {g : G} : IsComplement ({g} : Set G) univ :=
  ⟨fun ⟨⟨_, rfl⟩, _⟩ ⟨⟨_, rfl⟩, _⟩ h => Prod.ext rfl (Subtype.ext (mul_left_cancel h)), fun x =>
    ⟨⟨⟨g, rfl⟩, g⁻¹ * x, ⟨⟩⟩, mul_inv_cancel_left g x⟩⟩


@[to_additive]
theorem isComplement_singleton_left {g : G} : IsComplement {g} S ↔ S = univ := by
  refine
    ⟨fun h => top_le_iff.mp fun x _ => ?_, fun h => (congr_arg _ h).mpr isComplement_singleton_univ⟩
  /-
    G : Type u_1
    inst✝ : Group G
    S : Set G
    g : G
    h : Subgroup.IsComplement (Singleton.singleton g) S
    x : G
    x✝ : Membership.mem Top.top x
    ⊢ Membership.mem S x
  -/
  obtain ⟨⟨⟨z, rfl : z = g⟩, y, _⟩, hy⟩ := h.2 (g * x)
  /-
    case intro.mk.mk.mk
    G : Type u_1
    inst✝ : Group G
    S : Set G
    x : G
    x✝ : Membership.mem Top.top x
    z : G
    h : Subgroup.IsComplement (Singleton.singleton z) S
    y : G
    property✝ : Membership.mem S y
    hy : Eq ((fun x => HMul.hMul ↑x.1 ↑x.2) { fst := ⟨z, ⋯⟩, snd := ⟨y, property✝⟩ …
    ⊢ Membership.mem S x
  -/
  rwa [← mul_left_cancel hy]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem isComplement_singleton_right {g : G} : IsComplement S {g} ↔ S = univ := by
  refine
    ⟨fun h => top_le_iff.mp fun x _ => ?_, fun h => h ▸ isComplement_univ_singleton⟩
  /-
    G : Type u_1
    inst✝ : Group G
    S : Set G
    g : G
    h : Subgroup.IsComplement S (Singleton.singleton g)
    x : G
    x✝ : Membership.mem Top.top x
    ⊢ Membership.mem S x
  -/
  obtain ⟨y, hy⟩ := h.2 (x * g)
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    S : Set G
    g : G
    h : Subgroup.IsComplement S (Singleton.singleton g)
    x : G
    x✝ : Membership.mem Top.top x
    y : Prod ↑S ↑(Singleton.singleton g)
    hy : Eq ((fun x => HMul.hMul ↑x.1 ↑x.2) y) (HMul.hMul x g)
    ⊢ Membership.mem S x
  -/
  conv_rhs at hy => rw [← show y.2.1 = g from y.2.2]
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    S : Set G
    g : G
    h : Subgroup.IsComplement S (Singleton.singleton g)
    x : G
    x✝ : Membership.mem Top.top x
    y : Prod ↑S ↑(Singleton.singleton g)
    hy : Eq ((fun x => HMul.hMul ↑x.1 ↑x.2) y) (HMul.hMul x ↑y.2)
    ⊢ Membership.mem S x
  -/
  rw [← mul_right_cancel hy]
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    S : Set G
    g : G
    h : Subgroup.IsComplement S (Singleton.singleton g)
    x : G
    x✝ : Membership.mem Top.top x
    y : Prod ↑S ↑(Singleton.singleton g)
    hy : Eq ((fun x => HMul.hMul ↑x.1 ↑x.2) y) (HMul.hMul x ↑y.2)
    ⊢ Membership.mem S ↑y.1
  -/
  exact y.1.2
  /-
    🎉 no goals
  -/


@[to_additive]
theorem isComplement_univ_left : IsComplement univ S ↔ ∃ g : G, S = {g} := by
  refine
    ⟨fun h => Set.exists_eq_singleton_iff_nonempty_subsingleton.mpr ⟨?_, fun a ha b hb => ?_⟩, ?_⟩
    /-
      case refine_1
      G : Type u_1
      inst✝ : Group G
      S : Set G
      h : Subgroup.IsComplement Set.univ S
      ⊢ S.Nonempty
    -/
  · obtain ⟨a, _⟩ := h.2 1
    /-
      case refine_1.intro
      G : Type u_1
      inst✝ : Group G
      S : Set G
      h : Subgroup.IsComplement Set.univ S
      a : Prod ↑Set.univ ↑S
      h✝ : Eq ((fun x => HMul.hMul ↑x.1 ↑x.2) a) 1
      ⊢ S.Nonempty
    -/
    exact ⟨a.2.1, a.2.2⟩
    /-
      🎉 no goals
    -/
  · have : (⟨⟨_, mem_top a⁻¹⟩, ⟨a, ha⟩⟩ : (⊤ : Set G) × S) = ⟨⟨_, mem_top b⁻¹⟩, ⟨b, hb⟩⟩ :=
      h.1 ((inv_mul_cancel a).trans (inv_mul_cancel b).symm)
    /-
      case refine_2
      G : Type u_1
      inst✝ : Group G
      S : Set G
      h : Subgroup.IsComplement Set.univ S
      a : G
      ha : Membership.mem S a
      b : G
      hb : Membership.mem S b
      this : Eq { fst := ⟨Inv.inv a, ⋯⟩, snd := ⟨a, ha⟩ } { fst := ⟨Inv.inv b, ⋯⟩, s …
      ⊢ Eq a b
    -/
    exact Subtype.ext_iff.mp (Prod.ext_iff.mp this).2
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      G : Type u_1
      inst✝ : Group G
      S : Set G
      ⊢ (Exists fun g => Eq S (Singleton.singleton g)) → Subgroup.IsComplement Set.u …
    -/
  · rintro ⟨g, rfl⟩
    /-
      case refine_3.intro
      G : Type u_1
      inst✝ : Group G
      g : G
      ⊢ Subgroup.IsComplement Set.univ (Singleton.singleton g)
    -/
    exact isComplement_univ_singleton
    /-
      🎉 no goals
    -/


@[to_additive]
theorem isComplement_univ_right : IsComplement S univ ↔ ∃ g : G, S = {g} := by
  refine
    ⟨fun h => Set.exists_eq_singleton_iff_nonempty_subsingleton.mpr ⟨?_, fun a ha b hb => ?_⟩, ?_⟩
    /-
      case refine_1
      G : Type u_1
      inst✝ : Group G
      S : Set G
      h : Subgroup.IsComplement S Set.univ
      ⊢ S.Nonempty
    -/
  · obtain ⟨a, _⟩ := h.2 1
    /-
      case refine_1.intro
      G : Type u_1
      inst✝ : Group G
      S : Set G
      h : Subgroup.IsComplement S Set.univ
      a : Prod ↑S ↑Set.univ
      h✝ : Eq ((fun x => HMul.hMul ↑x.1 ↑x.2) a) 1
      ⊢ S.Nonempty
    -/
    exact ⟨a.1.1, a.1.2⟩
    /-
      🎉 no goals
    -/
  · have : (⟨⟨a, ha⟩, ⟨_, mem_top a⁻¹⟩⟩ : S × (⊤ : Set G)) = ⟨⟨b, hb⟩, ⟨_, mem_top b⁻¹⟩⟩ :=
      h.1 ((mul_inv_cancel a).trans (mul_inv_cancel b).symm)
    /-
      case refine_2
      G : Type u_1
      inst✝ : Group G
      S : Set G
      h : Subgroup.IsComplement S Set.univ
      a : G
      ha : Membership.mem S a
      b : G
      hb : Membership.mem S b
      this : Eq { fst := ⟨a, ha⟩, snd := ⟨Inv.inv a, ⋯⟩ } { fst := ⟨b, hb⟩, snd := ⟨ …
      ⊢ Eq a b
    -/
    exact Subtype.ext_iff.mp (Prod.ext_iff.mp this).1
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      G : Type u_1
      inst✝ : Group G
      S : Set G
      ⊢ (Exists fun g => Eq S (Singleton.singleton g)) → Subgroup.IsComplement S Set …
    -/
  · rintro ⟨g, rfl⟩
    /-
      case refine_3.intro
      G : Type u_1
      inst✝ : Group G
      g : G
      ⊢ Subgroup.IsComplement (Singleton.singleton g) Set.univ
    -/
    exact isComplement_singleton_univ
    /-
      🎉 no goals
    -/


@[to_additive]
lemma IsComplement.mul_eq (h : IsComplement S T) : S * T = univ :=
                               /-
                                 G : Type u_1
                                 inst✝ : Group G
                                 S T : Set G
                                 h : Subgroup.IsComplement S T
                                 x : G
                                 ⊢ Membership.mem (HMul.hMul S T) x
                               -/
  eq_univ_of_forall fun x ↦ by simpa [mem_mul] using (h.existsUnique x).exists
                               /-
                                 🎉 no goals
                               -/


@[to_additive AddSubgroup.IsComplement.card_mul_card]
lemma IsComplement.card_mul_card (h : IsComplement S T) : Nat.card S * Nat.card T = Nat.card G :=
  (Nat.card_prod _ _).symm.trans <| Nat.card_congr <| Equiv.ofBijective _ h


@[to_additive]
theorem isComplement'_top_bot : IsComplement' (⊤ : Subgroup G) ⊥ :=
  isComplement_univ_singleton


@[to_additive]
theorem isComplement'_bot_top : IsComplement' (⊥ : Subgroup G) ⊤ :=
  isComplement_singleton_univ


@[to_additive (attr := simp)]
theorem isComplement'_bot_left : IsComplement' ⊥ H ↔ H = ⊤ :=
  isComplement_singleton_left.trans coe_eq_univ


@[to_additive (attr := simp)]
theorem isComplement'_bot_right : IsComplement' H ⊥ ↔ H = ⊤ :=
  isComplement_singleton_right.trans coe_eq_univ


@[to_additive (attr := simp)]
theorem isComplement'_top_left : IsComplement' ⊤ H ↔ H = ⊥ :=
  isComplement_univ_left.trans coe_eq_singleton


@[to_additive (attr := simp)]
theorem isComplement'_top_right : IsComplement' H ⊤ ↔ H = ⊥ :=
  isComplement_univ_right.trans coe_eq_singleton


@[to_additive]
lemma isComplement_iff_existsUnique_inv_mul_mem :
    IsComplement S T ↔ ∀ g, ∃! s : S, (s : G)⁻¹ * g ∈ T := by
  /-
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    ⊢ Iff (Subgroup.IsComplement S T) (∀ (g : G), ExistsUnique fun s => Membership …
  -/
  convert isComplement_iff_existsUnique with g
  /-
    case h.e'_2.h.a
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    g : G
    ⊢ Iff (ExistsUnique fun s => Membership.mem T (HMul.hMul (Inv.inv ↑s) g)) (Exi …
  -/
  constructor <;> rintro ⟨x, hx, hx'⟩
    /-
      case h.e'_2.h.a.mp.intro.intro
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      g : G
      x : ↑S
      hx : Membership.mem T (HMul.hMul (Inv.inv ↑x) g)
      hx' : ∀ (y : ↑S), (fun s => Membership.mem T (HMul.hMul (Inv.inv ↑s) g)) y → E …
      ⊢ ExistsUnique fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g
    -/
  · exact ⟨(x, ⟨_, hx⟩), by simp, by aesop⟩
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.h.a.mpr.intro.intro
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      g : G
      x : Prod ↑S ↑T
      hx : Eq (HMul.hMul ↑x.1 ↑x.2) g
      hx' : ∀ (y : Prod ↑S ↑T), (fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g) y → Eq y x
      ⊢ ExistsUnique fun s => Membership.mem T (HMul.hMul (Inv.inv ↑s) g)
    -/
  · exact ⟨x.1, by simp [← hx], fun y hy ↦ (Prod.ext_iff.1 <| by simpa using hx' (y, ⟨_, hy⟩)).1⟩
    /-
      🎉 no goals
    -/


set_option linter.deprecated false in
@[to_additive
  (attr := deprecated isComplement_iff_existsUnique_inv_mul_mem (since := "2024-12-18"))]
theorem mem_leftTransversals_iff_existsUnique_inv_mul_mem :
    S ∈ leftTransversals T ↔ ∀ g : G, ∃! s : S, (s : G)⁻¹ * g ∈ T := by
  /-
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    ⊢ Iff (Membership.mem (Subgroup.leftTransversals T) S) (∀ (g : G), ExistsUniqu …
  -/
  rw [leftTransversals, Set.mem_setOf_eq, isComplement_iff_existsUnique]
  /-
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    ⊢ Iff (∀ (g : G), ExistsUnique fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g) (∀ (g : G) …
  -/
  refine ⟨fun h g => ?_, fun h g => ?_⟩
    /-
      case refine_1
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      h : ∀ (g : G), ExistsUnique fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g
      g : G
      ⊢ ExistsUnique fun s => Membership.mem T (HMul.hMul (Inv.inv ↑s) g)
    -/
  · obtain ⟨x, h1, h2⟩ := h g
    exact
      ⟨x.1, (congr_arg (· ∈ T) (eq_inv_mul_of_mul_eq h1)).mp x.2.2, fun y hy =>
        (Prod.ext_iff.mp (h2 ⟨y, (↑y)⁻¹ * g, hy⟩ (mul_inv_cancel_left ↑y g))).1⟩
    /-
      case refine_2
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      h : ∀ (g : G), ExistsUnique fun s => Membership.mem T (HMul.hMul (Inv.inv ↑s) g)
      g : G
      ⊢ ExistsUnique fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g
    -/
  · obtain ⟨x, h1, h2⟩ := h g
    /-
      case refine_2.intro.intro
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      h : ∀ (g : G), ExistsUnique fun s => Membership.mem T (HMul.hMul (Inv.inv ↑s) g)
      g : G
      x : ↑S
      h1 : Membership.mem T (HMul.hMul (Inv.inv ↑x) g)
      h2 : ∀ (y : ↑S), (fun s => Membership.mem T (HMul.hMul (Inv.inv ↑s) g)) y → Eq …
      ⊢ ExistsUnique fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g
    -/
    refine ⟨⟨x, (↑x)⁻¹ * g, h1⟩, mul_inv_cancel_left (↑x) g, fun y hy => ?_⟩
    /-
      case refine_2.intro.intro
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      h : ∀ (g : G), ExistsUnique fun s => Membership.mem T (HMul.hMul (Inv.inv ↑s) g)
      g : G
      x : ↑S
      h1 : Membership.mem T (HMul.hMul (Inv.inv ↑x) g)
      h2 : ∀ (y : ↑S), (fun s => Membership.mem T (HMul.hMul (Inv.inv ↑s) g)) y → Eq …
      y : Prod ↑S ↑T
      hy : (fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g) y
      ⊢ Eq y { fst := x, snd := ⟨HMul.hMul (Inv.inv ↑x) g, h1⟩ }
    -/
    have hf := h2 y.1 ((congr_arg (· ∈ T) (eq_inv_mul_of_mul_eq hy)).mp y.2.2)
    /-
      case refine_2.intro.intro
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      h : ∀ (g : G), ExistsUnique fun s => Membership.mem T (HMul.hMul (Inv.inv ↑s) g)
      g : G
      x : ↑S
      h1 : Membership.mem T (HMul.hMul (Inv.inv ↑x) g)
      h2 : ∀ (y : ↑S), (fun s => Membership.mem T (HMul.hMul (Inv.inv ↑s) g)) y → Eq …
      y : Prod ↑S ↑T
      hy : (fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g) y
      hf : Eq y.1 x
      ⊢ Eq y { fst := x, snd := ⟨HMul.hMul (Inv.inv ↑x) g, h1⟩ }
    -/
    exact Prod.ext hf (Subtype.ext (eq_inv_mul_of_mul_eq (hf ▸ hy)))
    /-
      🎉 no goals
    -/


@[to_additive]
lemma isComplement_iff_existsUnique_mul_inv_mem :
    IsComplement S T ↔ ∀ g, ∃! t : T, g * (t : G)⁻¹ ∈ S := by
  /-
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    ⊢ Iff (Subgroup.IsComplement S T) (∀ (g : G), ExistsUnique fun t => Membership …
  -/
  convert isComplement_iff_existsUnique with g
  /-
    case h.e'_2.h.a
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    g : G
    ⊢ Iff (ExistsUnique fun t => Membership.mem S (HMul.hMul g (Inv.inv ↑t))) (Exi …
  -/
  constructor <;> rintro ⟨x, hx, hx'⟩
    /-
      case h.e'_2.h.a.mp.intro.intro
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      g : G
      x : ↑T
      hx : Membership.mem S (HMul.hMul g (Inv.inv ↑x))
      hx' : ∀ (y : ↑T), (fun t => Membership.mem S (HMul.hMul g (Inv.inv ↑t))) y → E …
      ⊢ ExistsUnique fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g
    -/
  · exact ⟨(⟨_, hx⟩, x), by simp, by aesop⟩
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.h.a.mpr.intro.intro
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      g : G
      x : Prod ↑S ↑T
      hx : Eq (HMul.hMul ↑x.1 ↑x.2) g
      hx' : ∀ (y : Prod ↑S ↑T), (fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g) y → Eq y x
      ⊢ ExistsUnique fun t => Membership.mem S (HMul.hMul g (Inv.inv ↑t))
    -/
  · exact ⟨x.2, by simp [← hx], fun y hy ↦ (Prod.ext_iff.1 <| by simpa using hx' (⟨_, hy⟩, y)).2⟩
    /-
      🎉 no goals
    -/


set_option linter.deprecated false in
@[to_additive
  (attr := deprecated isComplement_iff_existsUnique_mul_inv_mem (since := "2024-12-18"))]
theorem mem_rightTransversals_iff_existsUnique_mul_inv_mem :
    S ∈ rightTransversals T ↔ ∀ g : G, ∃! s : S, g * (s : G)⁻¹ ∈ T := by
  /-
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    ⊢ Iff (Membership.mem (Subgroup.rightTransversals T) S) (∀ (g : G), ExistsUniq …
  -/
  rw [rightTransversals, Set.mem_setOf_eq, isComplement_iff_existsUnique]
  /-
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    ⊢ Iff (∀ (g : G), ExistsUnique fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g) (∀ (g : G) …
  -/
  refine ⟨fun h g => ?_, fun h g => ?_⟩
    /-
      case refine_1
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      h : ∀ (g : G), ExistsUnique fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g
      g : G
      ⊢ ExistsUnique fun s => Membership.mem T (HMul.hMul g (Inv.inv ↑s))
    -/
  · obtain ⟨x, h1, h2⟩ := h g
    exact
      ⟨x.2, (congr_arg (· ∈ T) (eq_mul_inv_of_mul_eq h1)).mp x.1.2, fun y hy =>
        (Prod.ext_iff.mp (h2 ⟨⟨g * (↑y)⁻¹, hy⟩, y⟩ (inv_mul_cancel_right g y))).2⟩
    /-
      case refine_2
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      h : ∀ (g : G), ExistsUnique fun s => Membership.mem T (HMul.hMul g (Inv.inv ↑s))
      g : G
      ⊢ ExistsUnique fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g
    -/
  · obtain ⟨x, h1, h2⟩ := h g
    /-
      case refine_2.intro.intro
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      h : ∀ (g : G), ExistsUnique fun s => Membership.mem T (HMul.hMul g (Inv.inv ↑s))
      g : G
      x : ↑S
      h1 : Membership.mem T (HMul.hMul g (Inv.inv ↑x))
      h2 : ∀ (y : ↑S), (fun s => Membership.mem T (HMul.hMul g (Inv.inv ↑s))) y → Eq …
      ⊢ ExistsUnique fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g
    -/
    refine ⟨⟨⟨g * (↑x)⁻¹, h1⟩, x⟩, inv_mul_cancel_right g x, fun y hy => ?_⟩
    /-
      case refine_2.intro.intro
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      h : ∀ (g : G), ExistsUnique fun s => Membership.mem T (HMul.hMul g (Inv.inv ↑s))
      g : G
      x : ↑S
      h1 : Membership.mem T (HMul.hMul g (Inv.inv ↑x))
      h2 : ∀ (y : ↑S), (fun s => Membership.mem T (HMul.hMul g (Inv.inv ↑s))) y → Eq …
      y : Prod ↑T ↑S
      hy : (fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g) y
      ⊢ Eq y { fst := ⟨HMul.hMul g (Inv.inv ↑x), h1⟩, snd := x }
    -/
    have hf := h2 y.2 ((congr_arg (· ∈ T) (eq_mul_inv_of_mul_eq hy)).mp y.1.2)
    /-
      case refine_2.intro.intro
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      h : ∀ (g : G), ExistsUnique fun s => Membership.mem T (HMul.hMul g (Inv.inv ↑s))
      g : G
      x : ↑S
      h1 : Membership.mem T (HMul.hMul g (Inv.inv ↑x))
      h2 : ∀ (y : ↑S), (fun s => Membership.mem T (HMul.hMul g (Inv.inv ↑s))) y → Eq …
      y : Prod ↑T ↑S
      hy : (fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g) y
      hf : Eq y.2 x
      ⊢ Eq y { fst := ⟨HMul.hMul g (Inv.inv ↑x), h1⟩, snd := x }
    -/
    exact Prod.ext (Subtype.ext (eq_mul_inv_of_mul_eq (hf ▸ hy))) hf
    /-
      🎉 no goals
    -/


@[to_additive]
lemma isComplement_subgroup_right_iff_existsUnique_quotientGroupMk :
    IsComplement S H ↔ ∀ q : G ⧸ H, ∃! s : S, QuotientGroup.mk s.1 = q := by
  simp_rw [isComplement_iff_existsUnique_inv_mul_mem, SetLike.mem_coe, ← QuotientGroup.eq,
    QuotientGroup.forall_mk]


set_option linter.deprecated false in
@[to_additive
  (attr := deprecated isComplement_subgroup_right_iff_existsUnique_quotientGroupMk
    (since := "2024-12-18"))]
theorem mem_leftTransversals_iff_existsUnique_quotient_mk''_eq :
    S ∈ leftTransversals (H : Set G) ↔
      ∀ q : Quotient (QuotientGroup.leftRel H), ∃! s : S, Quotient.mk'' s.1 = q := by
  simp_rw [mem_leftTransversals_iff_existsUnique_inv_mul_mem, SetLike.mem_coe, ←
    QuotientGroup.eq]
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    S : Set G
    ⊢ Iff (∀ (g : G), ExistsUnique fun s => Eq ↑↑s ↑g) (∀ (q : Quotient (QuotientG …
  -/
  exact ⟨fun h q => Quotient.inductionOn' q h, fun h g => h (Quotient.mk'' g)⟩
  /-
    🎉 no goals
  -/


set_option linter.docPrime false in
@[to_additive]
lemma isComplement_subgroup_left_iff_existsUnique_quotientMk'' :
    IsComplement H T ↔
      ∀ q : Quotient (QuotientGroup.rightRel H), ∃! t : T, Quotient.mk'' t.1 = q := by
  simp_rw [isComplement_iff_existsUnique_mul_inv_mem, SetLike.mem_coe,
    ← QuotientGroup.rightRel_apply, ← Quotient.eq'', Quotient.forall]


set_option linter.deprecated false in
@[to_additive
  (attr := deprecated isComplement_subgroup_left_iff_existsUnique_quotientMk''
    (since := "2024-12-18"))]
theorem mem_rightTransversals_iff_existsUnique_quotient_mk''_eq :
    S ∈ rightTransversals (H : Set G) ↔
      ∀ q : Quotient (QuotientGroup.rightRel H), ∃! s : S, Quotient.mk'' s.1 = q := by
  simp_rw [mem_rightTransversals_iff_existsUnique_mul_inv_mem, SetLike.mem_coe, ←
    QuotientGroup.rightRel_apply, ← Quotient.eq'']
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    S : Set G
    ⊢ Iff (∀ (g : G), ExistsUnique fun s => Eq (Quotient.mk'' ↑s) (Quotient.mk'' g …
  -/
  exact ⟨fun h q => Quotient.inductionOn' q h, fun h g => h (Quotient.mk'' g)⟩
  /-
    🎉 no goals
  -/


@[to_additive]
lemma isComplement_subgroup_right_iff_bijective :
    IsComplement S H ↔ Bijective (S.restrict (QuotientGroup.mk : G → G ⧸ H)) :=
  isComplement_subgroup_right_iff_existsUnique_quotientGroupMk.trans
    (bijective_iff_existsUnique (S.restrict QuotientGroup.mk)).symm


set_option linter.deprecated false in
@[to_additive
  (attr := deprecated isComplement_subgroup_right_iff_bijective (since := "2024-12-18"))]
theorem mem_leftTransversals_iff_bijective :
    S ∈ leftTransversals (H : Set G) ↔
      Function.Bijective (S.restrict (Quotient.mk'' : G → Quotient (QuotientGroup.leftRel H))) :=
  mem_leftTransversals_iff_existsUnique_quotient_mk''_eq.trans
    (Function.bijective_iff_existsUnique (S.restrict Quotient.mk'')).symm


@[to_additive]
lemma isComplement_subgroup_left_iff_bijective :
    IsComplement H T ↔
      Bijective (T.restrict (Quotient.mk'' : G → Quotient (QuotientGroup.rightRel H))) :=
  isComplement_subgroup_left_iff_existsUnique_quotientMk''.trans
    (bijective_iff_existsUnique (T.restrict Quotient.mk'')).symm


set_option linter.deprecated false in
@[to_additive
  (attr := deprecated isComplement_subgroup_left_iff_bijective (since := "2024-12-18"))]
theorem mem_rightTransversals_iff_bijective :
    S ∈ rightTransversals (H : Set G) ↔
      Function.Bijective (S.restrict (Quotient.mk'' : G → Quotient (QuotientGroup.rightRel H))) :=
  mem_rightTransversals_iff_existsUnique_quotient_mk''_eq.trans
    (Function.bijective_iff_existsUnique (S.restrict Quotient.mk'')).symm


@[to_additive]
lemma IsComplement.card_left (h : IsComplement S H) : Nat.card S = H.index :=
  Nat.card_congr <| .ofBijective _ <| isComplement_subgroup_right_iff_bijective.mp h


set_option linter.deprecated false in
@[to_additive (attr := deprecated IsComplement.card_left (since := "2024-12-18"))]
theorem card_left_transversal (h : S ∈ leftTransversals (H : Set G)) : Nat.card S = H.index :=
  Nat.card_congr <| Equiv.ofBijective _ <| mem_leftTransversals_iff_bijective.mp h


@[to_additive]
lemma IsComplement.card_right (h : IsComplement H T) : Nat.card T = H.index :=
  Nat.card_congr <| (Equiv.ofBijective _ <| isComplement_subgroup_left_iff_bijective.mp h).trans <|
    QuotientGroup.quotientRightRelEquivQuotientLeftRel H


set_option linter.deprecated false in
@[to_additive (attr := deprecated IsComplement.card_right (since := "2024-12-18"))]
theorem card_right_transversal (h : S ∈ rightTransversals (H : Set G)) : Nat.card S = H.index :=
  Nat.card_congr <|
    (Equiv.ofBijective _ <| mem_rightTransversals_iff_bijective.mp h).trans <|
      QuotientGroup.quotientRightRelEquivQuotientLeftRel H


@[to_additive]
lemma isComplement_range_left {f : G ⧸ H → G} (hf : ∀ q, ↑(f q) = q) :
    IsComplement (range f) H := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    f : HasQuotient.Quotient G H → G
    hf : ∀ (q : HasQuotient.Quotient G H), Eq (↑(f q)) q
    ⊢ Subgroup.IsComplement (Set.range f) ↑H
  -/
  rw [isComplement_subgroup_right_iff_bijective]
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    f : HasQuotient.Quotient G H → G
    hf : ∀ (q : HasQuotient.Quotient G H), Eq (↑(f q)) q
    ⊢ Function.Bijective ((Set.range f).restrict QuotientGroup.mk)
  -/
  refine ⟨?_, fun q ↦ ⟨⟨f q, q, rfl⟩, hf q⟩⟩
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    f : HasQuotient.Quotient G H → G
    hf : ∀ (q : HasQuotient.Quotient G H), Eq (↑(f q)) q
    ⊢ Function.Injective ((Set.range f).restrict QuotientGroup.mk)
  -/
  rintro ⟨-, q₁, rfl⟩ ⟨-, q₂, rfl⟩ h
  /-
    case mk.intro.mk.intro
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    f : HasQuotient.Quotient G H → G
    hf : ∀ (q : HasQuotient.Quotient G H), Eq (↑(f q)) q
    q₁ q₂ : HasQuotient.Quotient G H
    h : Eq ((Set.range f).restrict QuotientGroup.mk ⟨f q₁, ⋯⟩) ((Set.range f).rest …
    ⊢ Eq ⟨f q₁, ⋯⟩ ⟨f q₂, ⋯⟩
  -/
  exact Subtype.ext <| congr_arg f <| ((hf q₁).symm.trans h).trans (hf q₂)
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[to_additive (attr := deprecated isComplement_range_left (since := "2024-12-18"))]
theorem range_mem_leftTransversals {f : G ⧸ H → G} (hf : ∀ q, ↑(f q) = q) :
    Set.range f ∈ leftTransversals (H : Set G) :=
  mem_leftTransversals_iff_bijective.mpr
        /-
          G : Type u_1
          inst✝ : Group G
          H : Subgroup G
          f : HasQuotient.Quotient G H → G
          hf : ∀ (q : HasQuotient.Quotient G H), Eq (↑(f q)) q
          ⊢ Function.Injective ((Set.range f).restrict Quotient.mk'')
        -/
    ⟨by rintro ⟨-, q₁, rfl⟩ ⟨-, q₂, rfl⟩ h
        /-
          case mk.intro.mk.intro
          G : Type u_1
          inst✝ : Group G
          H : Subgroup G
          f : HasQuotient.Quotient G H → G
          hf : ∀ (q : HasQuotient.Quotient G H), Eq (↑(f q)) q
          q₁ q₂ : HasQuotient.Quotient G H
          h : Eq ((Set.range f).restrict Quotient.mk'' ⟨f q₁, ⋯⟩) ((Set.range f).restric …
          ⊢ Eq ⟨f q₁, ⋯⟩ ⟨f q₂, ⋯⟩
        -/
        exact Subtype.ext <| congr_arg f <| ((hf q₁).symm.trans h).trans (hf q₂),
        /-
          🎉 no goals
        -/
      fun q => ⟨⟨f q, q, rfl⟩, hf q⟩⟩


@[to_additive]
lemma isComplement_range_right {f : Quotient (QuotientGroup.rightRel H) → G}
    (hf : ∀ q, Quotient.mk'' (f q) = q) : IsComplement H (range f) := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    f : Quotient (QuotientGroup.rightRel H) → G
    hf : ∀ (q : Quotient (QuotientGroup.rightRel H)), Eq (Quotient.mk'' (f q)) q
    ⊢ Subgroup.IsComplement (↑H) (Set.range f)
  -/
  rw [isComplement_subgroup_left_iff_bijective]
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    f : Quotient (QuotientGroup.rightRel H) → G
    hf : ∀ (q : Quotient (QuotientGroup.rightRel H)), Eq (Quotient.mk'' (f q)) q
    ⊢ Function.Bijective ((Set.range f).restrict Quotient.mk'')
  -/
  refine ⟨?_, fun q ↦ ⟨⟨f q, q, rfl⟩, hf q⟩⟩
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    f : Quotient (QuotientGroup.rightRel H) → G
    hf : ∀ (q : Quotient (QuotientGroup.rightRel H)), Eq (Quotient.mk'' (f q)) q
    ⊢ Function.Injective ((Set.range f).restrict Quotient.mk'')
  -/
  rintro ⟨-, q₁, rfl⟩ ⟨-, q₂, rfl⟩ h
  /-
    case mk.intro.mk.intro
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    f : Quotient (QuotientGroup.rightRel H) → G
    hf : ∀ (q : Quotient (QuotientGroup.rightRel H)), Eq (Quotient.mk'' (f q)) q
    q₁ q₂ : Quotient (QuotientGroup.rightRel H)
    h : Eq ((Set.range f).restrict Quotient.mk'' ⟨f q₁, ⋯⟩) ((Set.range f).restric …
    ⊢ Eq ⟨f q₁, ⋯⟩ ⟨f q₂, ⋯⟩
  -/
  exact Subtype.ext <| congr_arg f <| ((hf q₁).symm.trans h).trans (hf q₂)
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[to_additive (attr := deprecated isComplement_range_right (since := "2024-12-18"))]
theorem range_mem_rightTransversals {f : Quotient (QuotientGroup.rightRel H) → G}
    (hf : ∀ q, Quotient.mk'' (f q) = q) : Set.range f ∈ rightTransversals (H : Set G) :=
  mem_rightTransversals_iff_bijective.mpr
        /-
          G : Type u_1
          inst✝ : Group G
          H : Subgroup G
          f : Quotient (QuotientGroup.rightRel H) → G
          hf : ∀ (q : Quotient (QuotientGroup.rightRel H)), Eq (Quotient.mk'' (f q)) q
          ⊢ Function.Injective ((Set.range f).restrict Quotient.mk'')
        -/
    ⟨by rintro ⟨-, q₁, rfl⟩ ⟨-, q₂, rfl⟩ h
        /-
          case mk.intro.mk.intro
          G : Type u_1
          inst✝ : Group G
          H : Subgroup G
          f : Quotient (QuotientGroup.rightRel H) → G
          hf : ∀ (q : Quotient (QuotientGroup.rightRel H)), Eq (Quotient.mk'' (f q)) q
          q₁ q₂ : Quotient (QuotientGroup.rightRel H)
          h : Eq ((Set.range f).restrict Quotient.mk'' ⟨f q₁, ⋯⟩) ((Set.range f).restric …
          ⊢ Eq ⟨f q₁, ⋯⟩ ⟨f q₂, ⋯⟩
        -/
        exact Subtype.ext <| congr_arg f <| ((hf q₁).symm.trans h).trans (hf q₂),
        /-
          🎉 no goals
        -/
      fun q => ⟨⟨f q, q, rfl⟩, hf q⟩⟩


@[to_additive]
lemma exists_isComplement_left (H : Subgroup G) (g : G) : ∃ S, IsComplement S H ∧ g ∈ S := by
  classical
  refine ⟨Set.range (Function.update Quotient.out _ g), isComplement_range_left fun q ↦ ?_,
    QuotientGroup.mk g, Function.update_self (Quotient.mk'' g) g Quotient.out⟩
  by_cases hq : q = Quotient.mk'' g
  · exact hq.symm ▸ congr_arg _ (Function.update_self (Quotient.mk'' g) g Quotient.out)
  · refine Function.update_of_ne ?_ g Quotient.out ▸ q.out_eq'
    exact hq


set_option linter.deprecated false in
@[to_additive (attr := deprecated exists_isComplement_left (since := "2024-12-18"))]
lemma exists_left_transversal (H : Subgroup G) (g : G) :
    ∃ S ∈ leftTransversals (H : Set G), g ∈ S := by
  classical
    refine
      ⟨Set.range (Function.update Quotient.out _ g), range_mem_leftTransversals fun q => ?_,
        Quotient.mk'' g, Function.update_self (Quotient.mk'' g) g Quotient.out⟩
    by_cases hq : q = Quotient.mk'' g
    · exact hq.symm ▸ congr_arg _ (Function.update_self (Quotient.mk'' g) g Quotient.out)
    · refine (Function.update_of_ne ?_ g Quotient.out) ▸ q.out_eq'
      exact hq


@[to_additive]
lemma exists_isComplement_right (H : Subgroup G) (g : G) :
    ∃ T, IsComplement H T ∧ g ∈ T := by
  classical
  refine ⟨Set.range (Function.update Quotient.out _ g), isComplement_range_right fun q ↦ ?_,
    Quotient.mk'' g, Function.update_self (Quotient.mk'' g) g Quotient.out⟩
  by_cases hq : q = Quotient.mk'' g
  · exact hq.symm ▸ congr_arg _ (Function.update_self (Quotient.mk'' g) g Quotient.out)
  · refine Function.update_of_ne ?_ g Quotient.out ▸ q.out_eq'
    exact hq


set_option linter.deprecated false in
@[to_additive (attr := deprecated exists_isComplement_right (since := "2024-12-18"))]
lemma exists_right_transversal (H : Subgroup G) (g : G) :
    ∃ S ∈ rightTransversals (H : Set G), g ∈ S := by
  classical
    refine
      ⟨Set.range (Function.update Quotient.out _ g), range_mem_rightTransversals fun q => ?_,
        Quotient.mk'' g, Function.update_self (Quotient.mk'' g) g Quotient.out⟩
    by_cases hq : q = Quotient.mk'' g
    · exact hq.symm ▸ congr_arg _ (Function.update_self (Quotient.mk'' g) g Quotient.out)
    · exact Eq.trans (congr_arg _ (Function.update_of_ne hq g Quotient.out)) q.out_eq'


/-- Given two subgroups `H' ⊆ H`, there exists a left transversal to `H'` inside `H`. -/
@[to_additive "Given two subgroups `H' ⊆ H`, there exists a transversal to `H'` inside `H`"]
lemma exists_left_transversal_of_le {H' H : Subgroup G} (h : H' ≤ H) :
    ∃ S : Set G, S * H' = H ∧ Nat.card S * Nat.card H' = Nat.card H := by
  /-
    G : Type u_1
    inst✝ : Group G
    H' H : Subgroup G
    h : LE.le H' H
    ⊢ Exists fun S => And (Eq (HMul.hMul S ↑H') ↑H) (Eq (HMul.hMul (Nat.card ↑S) ( …
  -/
  let H'' : Subgroup H := H'.comap H.subtype
  /-
    G : Type u_1
    inst✝ : Group G
    H' H : Subgroup G
    h : LE.le H' H
    H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
    ⊢ Exists fun S => And (Eq (HMul.hMul S ↑H') ↑H) (Eq (HMul.hMul (Nat.card ↑S) ( …
  -/
  have : H' = H''.map H.subtype := by simp [H'', h]
  /-
    G : Type u_1
    inst✝ : Group G
    H' H : Subgroup G
    h : LE.le H' H
    H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
    this : Eq H' (Subgroup.map H.subtype H'')
    ⊢ Exists fun S => And (Eq (HMul.hMul S ↑H') ↑H) (Eq (HMul.hMul (Nat.card ↑S) ( …
  -/
  rw [this]
  /-
    G : Type u_1
    inst✝ : Group G
    H' H : Subgroup G
    h : LE.le H' H
    H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
    this : Eq H' (Subgroup.map H.subtype H'')
    ⊢ Exists fun S => And (Eq (HMul.hMul S ↑(Subgroup.map H.subtype H'')) ↑H) (Eq  …
  -/
  obtain ⟨S, cmem, -⟩ := H''.exists_isComplement_left 1
  /-
    case intro.intro
    G : Type u_1
    inst✝ : Group G
    H' H : Subgroup G
    h : LE.le H' H
    H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
    this : Eq H' (Subgroup.map H.subtype H'')
    S : Set (Subtype fun x => Membership.mem H x)
    cmem : Subgroup.IsComplement S ↑H''
    ⊢ Exists fun S => And (Eq (HMul.hMul S ↑(Subgroup.map H.subtype H'')) ↑H) (Eq  …
  -/
  refine ⟨H.subtype '' S, ?_, ?_⟩
    /-
      case intro.intro.refine_1
      G : Type u_1
      inst✝ : Group G
      H' H : Subgroup G
      h : LE.le H' H
      H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
      this : Eq H' (Subgroup.map H.subtype H'')
      S : Set (Subtype fun x => Membership.mem H x)
      cmem : Subgroup.IsComplement S ↑H''
      ⊢ Eq (HMul.hMul (Set.image (⇑H.subtype) S) ↑(Subgroup.map H.subtype H'')) ↑H
    -/
  · have : H.subtype '' (S * H'') = H.subtype '' S * H''.map H.subtype := image_mul H.subtype
    /-
      case intro.intro.refine_1
      G : Type u_1
      inst✝ : Group G
      H' H : Subgroup G
      h : LE.le H' H
      H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
      this✝ : Eq H' (Subgroup.map H.subtype H'')
      S : Set (Subtype fun x => Membership.mem H x)
      cmem : Subgroup.IsComplement S ↑H''
      this : Eq (Set.image (⇑H.subtype) (HMul.hMul S ↑H'')) (HMul.hMul (Set.image (⇑ …
      ⊢ Eq (HMul.hMul (Set.image (⇑H.subtype) S) ↑(Subgroup.map H.subtype H'')) ↑H
    -/
    rw [← this, cmem.mul_eq]
    /-
      case intro.intro.refine_1
      G : Type u_1
      inst✝ : Group G
      H' H : Subgroup G
      h : LE.le H' H
      H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
      this✝ : Eq H' (Subgroup.map H.subtype H'')
      S : Set (Subtype fun x => Membership.mem H x)
      cmem : Subgroup.IsComplement S ↑H''
      this : Eq (Set.image (⇑H.subtype) (HMul.hMul S ↑H'')) (HMul.hMul (Set.image (⇑ …
      ⊢ Eq (Set.image (⇑H.subtype) Set.univ) ↑H
    -/
    simp [Set.ext_iff]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      G : Type u_1
      inst✝ : Group G
      H' H : Subgroup G
      h : LE.le H' H
      H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
      this : Eq H' (Subgroup.map H.subtype H'')
      S : Set (Subtype fun x => Membership.mem H x)
      cmem : Subgroup.IsComplement S ↑H''
      ⊢ Eq (HMul.hMul (Nat.card ↑(Set.image (⇑H.subtype) S)) (Nat.card (Subtype fun  …
    -/
  · rw [← cmem.card_mul_card]
    /-
      case intro.intro.refine_2
      G : Type u_1
      inst✝ : Group G
      H' H : Subgroup G
      h : LE.le H' H
      H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
      this : Eq H' (Subgroup.map H.subtype H'')
      S : Set (Subtype fun x => Membership.mem H x)
      cmem : Subgroup.IsComplement S ↑H''
      ⊢ Eq (HMul.hMul (Nat.card ↑(Set.image (⇑H.subtype) S)) (Nat.card (Subtype fun  …
    -/
    refine congr_arg₂ (· * ·) ?_ ?_ <;>
      /-
        case intro.intro.refine_2.refine_1
        G : Type u_1
        inst✝ : Group G
        H' H : Subgroup G
        h : LE.le H' H
        H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
        this : Eq H' (Subgroup.map H.subtype H'')
        S : Set (Subtype fun x => Membership.mem H x)
        cmem : Subgroup.IsComplement S ↑H''
        ⊢ Eq (Nat.card ↑(Set.image (⇑H.subtype) S)) (Nat.card ↑S)
      -/
      /-
        🎉 no goals
      -/
      exact Nat.card_congr (Equiv.Set.image _ _ <| subtype_injective H).symm
      /-
        🎉 no goals
      -/


/-- Given two subgroups `H' ⊆ H`, there exists a right transversal to `H'` inside `H`. -/
@[to_additive "Given two subgroups `H' ⊆ H`, there exists a transversal to `H'` inside `H`"]
lemma exists_right_transversal_of_le {H' H : Subgroup G} (h : H' ≤ H) :
    ∃ S : Set G, H' * S = H ∧ Nat.card H' * Nat.card S = Nat.card H := by
  /-
    G : Type u_1
    inst✝ : Group G
    H' H : Subgroup G
    h : LE.le H' H
    ⊢ Exists fun S => And (Eq (HMul.hMul (↑H') S) ↑H) (Eq (HMul.hMul (Nat.card (Su …
  -/
  let H'' : Subgroup H := H'.comap H.subtype
  /-
    G : Type u_1
    inst✝ : Group G
    H' H : Subgroup G
    h : LE.le H' H
    H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
    ⊢ Exists fun S => And (Eq (HMul.hMul (↑H') S) ↑H) (Eq (HMul.hMul (Nat.card (Su …
  -/
  have : H' = H''.map H.subtype := by simp [H'', h]
  /-
    G : Type u_1
    inst✝ : Group G
    H' H : Subgroup G
    h : LE.le H' H
    H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
    this : Eq H' (Subgroup.map H.subtype H'')
    ⊢ Exists fun S => And (Eq (HMul.hMul (↑H') S) ↑H) (Eq (HMul.hMul (Nat.card (Su …
  -/
  rw [this]
  /-
    G : Type u_1
    inst✝ : Group G
    H' H : Subgroup G
    h : LE.le H' H
    H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
    this : Eq H' (Subgroup.map H.subtype H'')
    ⊢ Exists fun S => And (Eq (HMul.hMul (↑(Subgroup.map H.subtype H'')) S) ↑H) (E …
  -/
  obtain ⟨S, cmem, -⟩ := H''.exists_isComplement_right 1
  /-
    case intro.intro
    G : Type u_1
    inst✝ : Group G
    H' H : Subgroup G
    h : LE.le H' H
    H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
    this : Eq H' (Subgroup.map H.subtype H'')
    S : Set (Subtype fun x => Membership.mem H x)
    cmem : Subgroup.IsComplement (↑H'') S
    ⊢ Exists fun S => And (Eq (HMul.hMul (↑(Subgroup.map H.subtype H'')) S) ↑H) (E …
  -/
  refine ⟨H.subtype '' S, ?_, ?_⟩
    /-
      case intro.intro.refine_1
      G : Type u_1
      inst✝ : Group G
      H' H : Subgroup G
      h : LE.le H' H
      H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
      this : Eq H' (Subgroup.map H.subtype H'')
      S : Set (Subtype fun x => Membership.mem H x)
      cmem : Subgroup.IsComplement (↑H'') S
      ⊢ Eq (HMul.hMul (↑(Subgroup.map H.subtype H'')) (Set.image (⇑H.subtype) S)) ↑H
    -/
  · have : H.subtype '' (H'' * S) = H''.map H.subtype * H.subtype '' S := image_mul H.subtype
    /-
      case intro.intro.refine_1
      G : Type u_1
      inst✝ : Group G
      H' H : Subgroup G
      h : LE.le H' H
      H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
      this✝ : Eq H' (Subgroup.map H.subtype H'')
      S : Set (Subtype fun x => Membership.mem H x)
      cmem : Subgroup.IsComplement (↑H'') S
      this : Eq (Set.image (⇑H.subtype) (HMul.hMul (↑H'') S)) (HMul.hMul (↑(Subgroup …
      ⊢ Eq (HMul.hMul (↑(Subgroup.map H.subtype H'')) (Set.image (⇑H.subtype) S)) ↑H
    -/
    rw [← this, cmem.mul_eq]
    /-
      case intro.intro.refine_1
      G : Type u_1
      inst✝ : Group G
      H' H : Subgroup G
      h : LE.le H' H
      H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
      this✝ : Eq H' (Subgroup.map H.subtype H'')
      S : Set (Subtype fun x => Membership.mem H x)
      cmem : Subgroup.IsComplement (↑H'') S
      this : Eq (Set.image (⇑H.subtype) (HMul.hMul (↑H'') S)) (HMul.hMul (↑(Subgroup …
      ⊢ Eq (Set.image (⇑H.subtype) Set.univ) ↑H
    -/
    simp [Set.ext_iff]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      G : Type u_1
      inst✝ : Group G
      H' H : Subgroup G
      h : LE.le H' H
      H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
      this : Eq H' (Subgroup.map H.subtype H'')
      S : Set (Subtype fun x => Membership.mem H x)
      cmem : Subgroup.IsComplement (↑H'') S
      ⊢ Eq (HMul.hMul (Nat.card (Subtype fun x => Membership.mem (Subgroup.map H.sub …
    -/
  · have : Nat.card H'' * Nat.card S = Nat.card H := cmem.card_mul_card
    /-
      case intro.intro.refine_2
      G : Type u_1
      inst✝ : Group G
      H' H : Subgroup G
      h : LE.le H' H
      H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
      this✝ : Eq H' (Subgroup.map H.subtype H'')
      S : Set (Subtype fun x => Membership.mem H x)
      cmem : Subgroup.IsComplement (↑H'') S
      this : Eq (HMul.hMul (Nat.card (Subtype fun x => Membership.mem H'' x)) (Nat.c …
      ⊢ Eq (HMul.hMul (Nat.card (Subtype fun x => Membership.mem (Subgroup.map H.sub …
    -/
    rw [← this]
    /-
      case intro.intro.refine_2
      G : Type u_1
      inst✝ : Group G
      H' H : Subgroup G
      h : LE.le H' H
      H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
      this✝ : Eq H' (Subgroup.map H.subtype H'')
      S : Set (Subtype fun x => Membership.mem H x)
      cmem : Subgroup.IsComplement (↑H'') S
      this : Eq (HMul.hMul (Nat.card (Subtype fun x => Membership.mem H'' x)) (Nat.c …
      ⊢ Eq (HMul.hMul (Nat.card (Subtype fun x => Membership.mem (Subgroup.map H.sub …
    -/
    refine congr_arg₂ (· * ·) ?_ ?_ <;>
      /-
        case intro.intro.refine_2.refine_1
        G : Type u_1
        inst✝ : Group G
        H' H : Subgroup G
        h : LE.le H' H
        H'' : Subgroup (Subtype fun x => Membership.mem H x) := Subgroup.comap H.subty …
        this✝ : Eq H' (Subgroup.map H.subtype H'')
        S : Set (Subtype fun x => Membership.mem H x)
        cmem : Subgroup.IsComplement (↑H'') S
        this : Eq (HMul.hMul (Nat.card (Subtype fun x => Membership.mem H'' x)) (Nat.c …
        ⊢ Eq (Nat.card (Subtype fun x => Membership.mem (Subgroup.map H.subtype H'') x …
      -/
      /-
        🎉 no goals
      -/
      exact Nat.card_congr (Equiv.Set.image _ _ <| subtype_injective H).symm
      /-
        🎉 no goals
      -/


/-- The equivalence `G ≃ S × T`, such that the inverse is  `(*) : S × T → G` -/
noncomputable def equiv {S T : Set G} (hST : IsComplement S T) : G ≃ S × T :=
  (Equiv.ofBijective (fun x : S × T => x.1.1 * x.2.1) hST).symm


@[simp] theorem equiv_symm_apply (x : S × T) : (hST.equiv.symm x : G) = x.1.1 * x.2.1 := rfl


@[simp]
theorem equiv_fst_mul_equiv_snd (g : G) : ↑(hST.equiv g).fst * (hST.equiv g).snd = g :=
  (Equiv.ofBijective (fun x : S × T => x.1.1 * x.2.1) hST).right_inv g


theorem equiv_fst_eq_mul_inv (g : G) : ↑(hST.equiv g).fst = g * ((hST.equiv g).snd : G)⁻¹ :=
  eq_mul_inv_of_mul_eq (hST.equiv_fst_mul_equiv_snd g)


theorem equiv_snd_eq_inv_mul (g : G) : ↑(hST.equiv g).snd = ((hST.equiv g).fst : G)⁻¹ * g :=
  eq_inv_mul_of_mul_eq (hST.equiv_fst_mul_equiv_snd g)


theorem equiv_fst_eq_iff_leftCosetEquivalence {g₁ g₂ : G} :
    (hSK.equiv g₁).fst = (hSK.equiv g₂).fst ↔ LeftCosetEquivalence K g₁ g₂ := by
  /-
    G : Type u_1
    inst✝ : Group G
    K : Subgroup G
    S : Set G
    hSK : Subgroup.IsComplement S ↑K
    g₁ g₂ : G
    ⊢ Iff (Eq (hSK.equiv g₁).1 (hSK.equiv g₂).1) (LeftCosetEquivalence (↑K) g₁ g₂)
  -/
  rw [LeftCosetEquivalence, leftCoset_eq_iff]
  /-
    G : Type u_1
    inst✝ : Group G
    K : Subgroup G
    S : Set G
    hSK : Subgroup.IsComplement S ↑K
    g₁ g₂ : G
    ⊢ Iff (Eq (hSK.equiv g₁).1 (hSK.equiv g₂).1) (Membership.mem K (HMul.hMul (Inv …
  -/
  constructor
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      K : Subgroup G
      S : Set G
      hSK : Subgroup.IsComplement S ↑K
      g₁ g₂ : G
      ⊢ Eq (hSK.equiv g₁).1 (hSK.equiv g₂).1 → Membership.mem K (HMul.hMul (Inv.inv  …
    -/
  · intro h
    rw [← hSK.equiv_fst_mul_equiv_snd g₂, ← hSK.equiv_fst_mul_equiv_snd g₁, ← h,
      mul_inv_rev, ← mul_assoc, inv_mul_cancel_right, ← coe_inv, ← coe_mul]
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      K : Subgroup G
      S : Set G
      hSK : Subgroup.IsComplement S ↑K
      g₁ g₂ : G
      h : Eq (hSK.equiv g₁).1 (hSK.equiv g₂).1
      ⊢ Membership.mem K ↑(HMul.hMul (Inv.inv (hSK.equiv g₁).2) (hSK.equiv g₂).2)
    -/
    exact Subtype.property _
    /-
      🎉 no goals
    -/
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      K : Subgroup G
      S : Set G
      hSK : Subgroup.IsComplement S ↑K
      g₁ g₂ : G
      ⊢ Membership.mem K (HMul.hMul (Inv.inv g₁) g₂) → Eq (hSK.equiv g₁).1 (hSK.equi …
    -/
  · intro h
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      K : Subgroup G
      S : Set G
      hSK : Subgroup.IsComplement S ↑K
      g₁ g₂ : G
      h : Membership.mem K (HMul.hMul (Inv.inv g₁) g₂)
      ⊢ Eq (hSK.equiv g₁).1 (hSK.equiv g₂).1
    -/
    apply (isComplement_iff_existsUnique_inv_mul_mem.1 hSK g₁).unique
    · -- This used to be `simp [...]` before https://github.com/leanprover/lean4/pull/2644
      /-
        case mpr.py₁
        G : Type u_1
        inst✝ : Group G
        K : Subgroup G
        S : Set G
        hSK : Subgroup.IsComplement S ↑K
        g₁ g₂ : G
        h : Membership.mem K (HMul.hMul (Inv.inv g₁) g₂)
        ⊢ Membership.mem (↑K) (HMul.hMul (Inv.inv ↑(hSK.equiv g₁).1) g₁)
      -/
      rw [equiv_fst_eq_mul_inv]; simp
                                 /-
                                   🎉 no goals
                                 -/
      /-
        case mpr.py₂
        G : Type u_1
        inst✝ : Group G
        K : Subgroup G
        S : Set G
        hSK : Subgroup.IsComplement S ↑K
        g₁ g₂ : G
        h : Membership.mem K (HMul.hMul (Inv.inv g₁) g₂)
        ⊢ Membership.mem (↑K) (HMul.hMul (Inv.inv ↑(hSK.equiv g₂).1) g₁)
      -/
    · rw [SetLike.mem_coe, ← mul_mem_cancel_right h]
      -- This used to be `simp [...]` before https://github.com/leanprover/lean4/pull/2644
      /-
        case mpr.py₂
        G : Type u_1
        inst✝ : Group G
        K : Subgroup G
        S : Set G
        hSK : Subgroup.IsComplement S ↑K
        g₁ g₂ : G
        h : Membership.mem K (HMul.hMul (Inv.inv g₁) g₂)
        ⊢ Membership.mem K (HMul.hMul (HMul.hMul (Inv.inv ↑(hSK.equiv g₂).1) g₁) (HMul …
      -/
      rw [equiv_fst_eq_mul_inv]; simp [equiv_fst_eq_mul_inv, ← mul_assoc]
                                 /-
                                   🎉 no goals
                                 -/


theorem equiv_snd_eq_iff_rightCosetEquivalence {g₁ g₂ : G} :
    (hHT.equiv g₁).snd = (hHT.equiv g₂).snd ↔ RightCosetEquivalence H g₁ g₂ := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    T : Set G
    hHT : Subgroup.IsComplement (↑H) T
    g₁ g₂ : G
    ⊢ Iff (Eq (hHT.equiv g₁).2 (hHT.equiv g₂).2) (RightCosetEquivalence (↑H) g₁ g₂)
  -/
  rw [RightCosetEquivalence, rightCoset_eq_iff]
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    T : Set G
    hHT : Subgroup.IsComplement (↑H) T
    g₁ g₂ : G
    ⊢ Iff (Eq (hHT.equiv g₁).2 (hHT.equiv g₂).2) (Membership.mem H (HMul.hMul g₂ ( …
  -/
  constructor
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      T : Set G
      hHT : Subgroup.IsComplement (↑H) T
      g₁ g₂ : G
      ⊢ Eq (hHT.equiv g₁).2 (hHT.equiv g₂).2 → Membership.mem H (HMul.hMul g₂ (Inv.i …
    -/
  · intro h
    rw [← hHT.equiv_fst_mul_equiv_snd g₂, ← hHT.equiv_fst_mul_equiv_snd g₁, ← h,
      mul_inv_rev, mul_assoc, mul_inv_cancel_left, ← coe_inv, ← coe_mul]
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      T : Set G
      hHT : Subgroup.IsComplement (↑H) T
      g₁ g₂ : G
      h : Eq (hHT.equiv g₁).2 (hHT.equiv g₂).2
      ⊢ Membership.mem H ↑(HMul.hMul (hHT.equiv g₂).1 (Inv.inv (hHT.equiv g₁).1))
    -/
    exact Subtype.property _
    /-
      🎉 no goals
    -/
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      T : Set G
      hHT : Subgroup.IsComplement (↑H) T
      g₁ g₂ : G
      ⊢ Membership.mem H (HMul.hMul g₂ (Inv.inv g₁)) → Eq (hHT.equiv g₁).2 (hHT.equi …
    -/
  · intro h
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      T : Set G
      hHT : Subgroup.IsComplement (↑H) T
      g₁ g₂ : G
      h : Membership.mem H (HMul.hMul g₂ (Inv.inv g₁))
      ⊢ Eq (hHT.equiv g₁).2 (hHT.equiv g₂).2
    -/
    apply (isComplement_iff_existsUnique_mul_inv_mem.1 hHT g₁).unique
    · -- This used to be `simp [...]` before https://github.com/leanprover/lean4/pull/2644
      /-
        case mpr.py₁
        G : Type u_1
        inst✝ : Group G
        H : Subgroup G
        T : Set G
        hHT : Subgroup.IsComplement (↑H) T
        g₁ g₂ : G
        h : Membership.mem H (HMul.hMul g₂ (Inv.inv g₁))
        ⊢ Membership.mem (↑H) (HMul.hMul g₁ (Inv.inv ↑(hHT.equiv g₁).2))
      -/
      rw [equiv_snd_eq_inv_mul]; simp
                                 /-
                                   🎉 no goals
                                 -/
      /-
        case mpr.py₂
        G : Type u_1
        inst✝ : Group G
        H : Subgroup G
        T : Set G
        hHT : Subgroup.IsComplement (↑H) T
        g₁ g₂ : G
        h : Membership.mem H (HMul.hMul g₂ (Inv.inv g₁))
        ⊢ Membership.mem (↑H) (HMul.hMul g₁ (Inv.inv ↑(hHT.equiv g₂).2))
      -/
    · rw [SetLike.mem_coe, ← mul_mem_cancel_left h]
      -- This used to be `simp [...]` before https://github.com/leanprover/lean4/pull/2644
      /-
        case mpr.py₂
        G : Type u_1
        inst✝ : Group G
        H : Subgroup G
        T : Set G
        hHT : Subgroup.IsComplement (↑H) T
        g₁ g₂ : G
        h : Membership.mem H (HMul.hMul g₂ (Inv.inv g₁))
        ⊢ Membership.mem H (HMul.hMul (HMul.hMul g₂ (Inv.inv g₁)) (HMul.hMul g₁ (Inv.i …
      -/
      rw [equiv_snd_eq_inv_mul, mul_assoc]; simp
                                            /-
                                              🎉 no goals
                                            -/


theorem leftCosetEquivalence_equiv_fst (g : G) :
    LeftCosetEquivalence K g ((hSK.equiv g).fst : G) := by
  -- This used to be `simp [...]` before https://github.com/leanprover/lean4/pull/2644
  /-
    G : Type u_1
    inst✝ : Group G
    K : Subgroup G
    S : Set G
    hSK : Subgroup.IsComplement S ↑K
    g : G
    ⊢ LeftCosetEquivalence (↑K) g ↑(hSK.equiv g).1
  -/
  rw [equiv_fst_eq_mul_inv]; simp [LeftCosetEquivalence, leftCoset_eq_iff]
                             /-
                               🎉 no goals
                             -/


theorem rightCosetEquivalence_equiv_snd (g : G) :
    RightCosetEquivalence H g ((hHT.equiv g).snd : G) := by
  -- This used to be `simp [...]` before https://github.com/leanprover/lean4/pull/2644
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    T : Set G
    hHT : Subgroup.IsComplement (↑H) T
    g : G
    ⊢ RightCosetEquivalence (↑H) g ↑(hHT.equiv g).2
  -/
  rw [RightCosetEquivalence, rightCoset_eq_iff, equiv_snd_eq_inv_mul]; simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem equiv_fst_eq_self_of_mem_of_one_mem {g : G} (h1 : 1 ∈ T) (hg : g ∈ S) :
    (hST.equiv g).fst = ⟨g, hg⟩ := by
  have : hST.equiv.symm (⟨g, hg⟩, ⟨1, h1⟩) = g := by
    rw [equiv, Equiv.ofBijective]; simp
  /-
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    hST : Subgroup.IsComplement S T
    g : G
    h1 : Membership.mem T 1
    hg : Membership.mem S g
    this : Eq (hST.equiv.symm { fst := ⟨g, hg⟩, snd := ⟨1, h1⟩ }) g
    ⊢ Eq (hST.equiv g).1 ⟨g, hg⟩
  -/
  conv_lhs => rw [← this, Equiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


theorem equiv_snd_eq_self_of_mem_of_one_mem {g : G} (h1 : 1 ∈ S) (hg : g ∈ T) :
    (hST.equiv g).snd = ⟨g, hg⟩ := by
  have : hST.equiv.symm (⟨1, h1⟩, ⟨g, hg⟩) = g := by
    rw [equiv, Equiv.ofBijective]; simp
  /-
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    hST : Subgroup.IsComplement S T
    g : G
    h1 : Membership.mem S 1
    hg : Membership.mem T g
    this : Eq (hST.equiv.symm { fst := ⟨1, h1⟩, snd := ⟨g, hg⟩ }) g
    ⊢ Eq (hST.equiv g).2 ⟨g, hg⟩
  -/
  conv_lhs => rw [← this, Equiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


theorem equiv_snd_eq_one_of_mem_of_one_mem {g : G} (h1 : 1 ∈ T) (hg : g ∈ S) :
    (hST.equiv g).snd = ⟨1, h1⟩ := by
  /-
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    hST : Subgroup.IsComplement S T
    g : G
    h1 : Membership.mem T 1
    hg : Membership.mem S g
    ⊢ Eq (hST.equiv g).2 ⟨1, h1⟩
  -/
  ext
  /-
    case a
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    hST : Subgroup.IsComplement S T
    g : G
    h1 : Membership.mem T 1
    hg : Membership.mem S g
    ⊢ Eq ↑(hST.equiv g).2 ↑⟨1, h1⟩
  -/
  rw [equiv_snd_eq_inv_mul, equiv_fst_eq_self_of_mem_of_one_mem _ h1 hg, inv_mul_cancel]
  /-
    🎉 no goals
  -/


theorem equiv_fst_eq_one_of_mem_of_one_mem {g : G} (h1 : 1 ∈ S) (hg : g ∈ T) :
    (hST.equiv g).fst = ⟨1, h1⟩ := by
  /-
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    hST : Subgroup.IsComplement S T
    g : G
    h1 : Membership.mem S 1
    hg : Membership.mem T g
    ⊢ Eq (hST.equiv g).1 ⟨1, h1⟩
  -/
  ext
  /-
    case a
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    hST : Subgroup.IsComplement S T
    g : G
    h1 : Membership.mem S 1
    hg : Membership.mem T g
    ⊢ Eq ↑(hST.equiv g).1 ↑⟨1, h1⟩
  -/
  rw [equiv_fst_eq_mul_inv, equiv_snd_eq_self_of_mem_of_one_mem _ h1 hg, mul_inv_cancel]
  /-
    🎉 no goals
  -/

-- This lemma has always been bad, but the linter only noticed after https://github.com/leanprover/lean4/pull/2644.

@[simp, nolint simpNF]
theorem equiv_mul_right (g : G) (k : K) :
    hSK.equiv (g * k) = ((hSK.equiv g).fst, (hSK.equiv g).snd * k) := by
  have : (hSK.equiv (g * k)).fst = (hSK.equiv g).fst :=
    hSK.equiv_fst_eq_iff_leftCosetEquivalence.2
      (by simp [LeftCosetEquivalence, leftCoset_eq_iff])
  /-
    G : Type u_1
    inst✝ : Group G
    K : Subgroup G
    S : Set G
    hSK : Subgroup.IsComplement S ↑K
    g : G
    k : Subtype fun x => Membership.mem K x
    this : Eq (hSK.equiv (HMul.hMul g ↑k)).1 (hSK.equiv g).1
    ⊢ Eq (hSK.equiv (HMul.hMul g ↑k)) { fst := (hSK.equiv g).1, snd := HMul.hMul ( …
  -/
  ext
    /-
      case fst.a
      G : Type u_1
      inst✝ : Group G
      K : Subgroup G
      S : Set G
      hSK : Subgroup.IsComplement S ↑K
      g : G
      k : Subtype fun x => Membership.mem K x
      this : Eq (hSK.equiv (HMul.hMul g ↑k)).1 (hSK.equiv g).1
      ⊢ Eq ↑(hSK.equiv (HMul.hMul g ↑k)).1 ↑{ fst := (hSK.equiv g).1, snd := HMul.hM …
    -/
  · rw [this]
    /-
      🎉 no goals
    -/
    /-
      case snd.a
      G : Type u_1
      inst✝ : Group G
      K : Subgroup G
      S : Set G
      hSK : Subgroup.IsComplement S ↑K
      g : G
      k : Subtype fun x => Membership.mem K x
      this : Eq (hSK.equiv (HMul.hMul g ↑k)).1 (hSK.equiv g).1
      ⊢ Eq ↑(hSK.equiv (HMul.hMul g ↑k)).2 ↑{ fst := (hSK.equiv g).1, snd := HMul.hM …
    -/
  · rw [coe_mul, equiv_snd_eq_inv_mul, this, equiv_snd_eq_inv_mul, mul_assoc]
    /-
      🎉 no goals
    -/


theorem equiv_mul_right_of_mem {g k : G} (h : k ∈ K) :
    hSK.equiv (g * k) = ((hSK.equiv g).fst, (hSK.equiv g).snd * ⟨k, h⟩) :=
  equiv_mul_right _ g ⟨k, h⟩

-- This lemma has always been bad, but the linter only noticed after https://github.com/leanprover/lean4/pull/2644.

@[simp, nolint simpNF]
theorem equiv_mul_left (h : H) (g : G) :
    hHT.equiv (h * g) = (h * (hHT.equiv g).fst, (hHT.equiv g).snd) := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    T : Set G
    hHT : Subgroup.IsComplement (↑H) T
    h : Subtype fun x => Membership.mem H x
    g : G
    ⊢ Eq (hHT.equiv (HMul.hMul (↑h) g)) { fst := HMul.hMul h (hHT.equiv g).1, snd  …
  -/
  have : (hHT.equiv (h * g)).2 = (hHT.equiv g).2 := hHT.equiv_snd_eq_iff_rightCosetEquivalence.2 ?_
    /-
      case refine_2
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      T : Set G
      hHT : Subgroup.IsComplement (↑H) T
      h : Subtype fun x => Membership.mem H x
      g : G
      this : Eq (hHT.equiv (HMul.hMul (↑h) g)).2 (hHT.equiv g).2
      ⊢ Eq (hHT.equiv (HMul.hMul (↑h) g)) { fst := HMul.hMul h (hHT.equiv g).1, snd  …
    -/
  · ext
      /-
        case refine_2.fst.a
        G : Type u_1
        inst✝ : Group G
        H : Subgroup G
        T : Set G
        hHT : Subgroup.IsComplement (↑H) T
        h : Subtype fun x => Membership.mem H x
        g : G
        this : Eq (hHT.equiv (HMul.hMul (↑h) g)).2 (hHT.equiv g).2
        ⊢ Eq ↑(hHT.equiv (HMul.hMul (↑h) g)).1 ↑{ fst := HMul.hMul h (hHT.equiv g).1,  …
      -/
    · rw [coe_mul, equiv_fst_eq_mul_inv, this, equiv_fst_eq_mul_inv, mul_assoc]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.snd.a
        G : Type u_1
        inst✝ : Group G
        H : Subgroup G
        T : Set G
        hHT : Subgroup.IsComplement (↑H) T
        h : Subtype fun x => Membership.mem H x
        g : G
        this : Eq (hHT.equiv (HMul.hMul (↑h) g)).2 (hHT.equiv g).2
        ⊢ Eq ↑(hHT.equiv (HMul.hMul (↑h) g)).2 ↑{ fst := HMul.hMul h (hHT.equiv g).1,  …
      -/
    · rw [this]
      /-
        🎉 no goals
      -/
    /-
      case refine_1
      G : Type u_1
      inst✝ : Group G
      H : Subgroup G
      T : Set G
      hHT : Subgroup.IsComplement (↑H) T
      h : Subtype fun x => Membership.mem H x
      g : G
      ⊢ RightCosetEquivalence (↑H) (HMul.hMul (↑h) g) g
    -/
  · simp [RightCosetEquivalence, ← smul_smul]
    /-
      🎉 no goals
    -/


theorem equiv_mul_left_of_mem {h g : G} (hh : h ∈ H) :
    hHT.equiv (h * g) = (⟨h, hh⟩ * (hHT.equiv g).fst, (hHT.equiv g).snd) :=
  equiv_mul_left _ ⟨h, hh⟩ g


theorem equiv_one (hs1 : 1 ∈ S) (ht1 : 1 ∈ T) :
    hST.equiv 1 = (⟨1, hs1⟩, ⟨1, ht1⟩) := by
  /-
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    hST : Subgroup.IsComplement S T
    hs1 : Membership.mem S 1
    ht1 : Membership.mem T 1
    ⊢ Eq (hST.equiv 1) { fst := ⟨1, hs1⟩, snd := ⟨1, ht1⟩ }
  -/
  rw [Equiv.apply_eq_iff_eq_symm_apply]; simp [equiv]
                                         /-
                                           🎉 no goals
                                         -/


theorem equiv_fst_eq_self_iff_mem {g : G} (h1 : 1 ∈ T) :
    ((hST.equiv g).fst : G) = g ↔ g ∈ S := by
  /-
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    hST : Subgroup.IsComplement S T
    g : G
    h1 : Membership.mem T 1
    ⊢ Iff (Eq (↑(hST.equiv g).1) g) (Membership.mem S g)
  -/
  constructor
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      hST : Subgroup.IsComplement S T
      g : G
      h1 : Membership.mem T 1
      ⊢ Eq (↑(hST.equiv g).1) g → Membership.mem S g
    -/
  · intro h
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      hST : Subgroup.IsComplement S T
      g : G
      h1 : Membership.mem T 1
      h : Eq (↑(hST.equiv g).1) g
      ⊢ Membership.mem S g
    -/
    rw [← h]
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      hST : Subgroup.IsComplement S T
      g : G
      h1 : Membership.mem T 1
      h : Eq (↑(hST.equiv g).1) g
      ⊢ Membership.mem S ↑(hST.equiv g).1
    -/
    exact Subtype.prop _
    /-
      🎉 no goals
    -/
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      hST : Subgroup.IsComplement S T
      g : G
      h1 : Membership.mem T 1
      ⊢ Membership.mem S g → Eq (↑(hST.equiv g).1) g
    -/
  · intro h
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      hST : Subgroup.IsComplement S T
      g : G
      h1 : Membership.mem T 1
      h : Membership.mem S g
      ⊢ Eq (↑(hST.equiv g).1) g
    -/
    rw [hST.equiv_fst_eq_self_of_mem_of_one_mem h1 h]
    /-
      🎉 no goals
    -/


theorem equiv_snd_eq_self_iff_mem {g : G} (h1 : 1 ∈ S) :
    ((hST.equiv g).snd : G) = g ↔ g ∈ T := by
  /-
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    hST : Subgroup.IsComplement S T
    g : G
    h1 : Membership.mem S 1
    ⊢ Iff (Eq (↑(hST.equiv g).2) g) (Membership.mem T g)
  -/
  constructor
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      hST : Subgroup.IsComplement S T
      g : G
      h1 : Membership.mem S 1
      ⊢ Eq (↑(hST.equiv g).2) g → Membership.mem T g
    -/
  · intro h
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      hST : Subgroup.IsComplement S T
      g : G
      h1 : Membership.mem S 1
      h : Eq (↑(hST.equiv g).2) g
      ⊢ Membership.mem T g
    -/
    rw [← h]
    /-
      case mp
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      hST : Subgroup.IsComplement S T
      g : G
      h1 : Membership.mem S 1
      h : Eq (↑(hST.equiv g).2) g
      ⊢ Membership.mem T ↑(hST.equiv g).2
    -/
    exact Subtype.prop _
    /-
      🎉 no goals
    -/
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      hST : Subgroup.IsComplement S T
      g : G
      h1 : Membership.mem S 1
      ⊢ Membership.mem T g → Eq (↑(hST.equiv g).2) g
    -/
  · intro h
    /-
      case mpr
      G : Type u_1
      inst✝ : Group G
      S T : Set G
      hST : Subgroup.IsComplement S T
      g : G
      h1 : Membership.mem S 1
      h : Membership.mem T g
      ⊢ Eq (↑(hST.equiv g).2) g
    -/
    rw [hST.equiv_snd_eq_self_of_mem_of_one_mem h1 h]
    /-
      🎉 no goals
    -/


theorem coe_equiv_fst_eq_one_iff_mem {g : G} (h1 : 1 ∈ S) :
    ((hST.equiv g).fst : G) = 1 ↔ g ∈ T := by
  /-
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    hST : Subgroup.IsComplement S T
    g : G
    h1 : Membership.mem S 1
    ⊢ Iff (Eq (↑(hST.equiv g).1) 1) (Membership.mem T g)
  -/
  rw [equiv_fst_eq_mul_inv, mul_inv_eq_one, eq_comm, equiv_snd_eq_self_iff_mem _ h1]
  /-
    🎉 no goals
  -/


theorem coe_equiv_snd_eq_one_iff_mem {g : G} (h1 : 1 ∈ T) :
    ((hST.equiv g).snd : G) = 1 ↔ g ∈ S := by
  /-
    G : Type u_1
    inst✝ : Group G
    S T : Set G
    hST : Subgroup.IsComplement S T
    g : G
    h1 : Membership.mem T 1
    ⊢ Iff (Eq (↑(hST.equiv g).2) 1) (Membership.mem S g)
  -/
  rw [equiv_snd_eq_inv_mul, inv_mul_eq_one, equiv_fst_eq_self_iff_mem _ h1]
  /-
    🎉 no goals
  -/


/-- A left transversal is in bijection with left cosets. -/
@[to_additive "A left transversal is in bijection with left cosets."]
noncomputable def leftQuotientEquiv (hS : IsComplement S H) : G ⧸ H ≃ S :=
  (Equiv.ofBijective _ (isComplement_subgroup_right_iff_bijective.mp hS)).symm


@[deprecated (since := "2024-12-28")]
alias _root_.Subgroup.MemLeftTransversals.toEquiv := leftQuotientEquiv


/-- A left transversal is finite iff the subgroup has finite index.-/
@[to_additive "A left transversal is finite iff the subgroup has finite index."]
theorem finite_left_iff (h : IsComplement S H) : Finite S ↔ H.FiniteIndex := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    S : Set G
    h : Subgroup.IsComplement S ↑H
    ⊢ Iff (Finite ↑S) H.FiniteIndex
  -/
  rw [← h.leftQuotientEquiv.finite_iff]
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    S : Set G
    h : Subgroup.IsComplement S ↑H
    ⊢ Iff (Finite (HasQuotient.Quotient G H)) H.FiniteIndex
  -/
  exact ⟨fun _ ↦ finiteIndex_of_finite_quotient H, fun _ ↦ finite_quotient_of_finiteIndex H⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-28")]
alias _root_.Subgroup.MemLeftTransversals.finite_iff := finite_left_iff


@[to_additive]
theorem quotientGroupMk_leftQuotientEquiv (hS : IsComplement S H) (q : G ⧸ H) :
    Quotient.mk'' (leftQuotientEquiv hS q : G) = q :=
  hS.leftQuotientEquiv.symm_apply_apply q


@[deprecated (since := "2024-12-28")]
alias _root_.Subgroup.MemLeftTransversals.mk''_toEquiv := quotientGroupMk_leftQuotientEquiv


@[to_additive]
theorem leftQuotientEquiv_apply {f : G ⧸ H → G} (hf : ∀ q, (f q : G ⧸ H) = q) (q : G ⧸ H) :
    (leftQuotientEquiv (isComplement_range_left hf) q : G) = f q := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    f : HasQuotient.Quotient G H → G
    hf : ∀ (q : HasQuotient.Quotient G H), Eq (↑(f q)) q
    q : HasQuotient.Quotient G H
    ⊢ Eq (↑(⋯.leftQuotientEquiv q)) (f q)
  -/
  refine (Subtype.ext_iff.mp ?_).trans (Subtype.coe_mk (f q) ⟨q, rfl⟩)
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    f : HasQuotient.Quotient G H → G
    hf : ∀ (q : HasQuotient.Quotient G H), Eq (↑(f q)) q
    q : HasQuotient.Quotient G H
    ⊢ Eq (⋯.leftQuotientEquiv q) ⟨f q, ⋯⟩
  -/
  exact (leftQuotientEquiv (isComplement_range_left hf)).apply_eq_iff_eq_symm_apply.mpr (hf q).symm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-28")]
alias _root_.Subgroup.MemLeftTransversals.toEquiv_apply := leftQuotientEquiv_apply


/-- A left transversal can be viewed as a function mapping each element of the group
  to the chosen representative from that left coset. -/
@[to_additive "A left transversal can be viewed as a function mapping each element of the group
  to the chosen representative from that left coset."]
noncomputable def toLeftFun (hS : IsComplement S H) : G → S := leftQuotientEquiv hS ∘ Quotient.mk''


@[deprecated (since := "2024-12-28")]
alias _root_.Subgroup.MemLeftTransversals.toFun := toLeftFun


@[to_additive]
theorem inv_toLeftFun_mul_mem (hS : IsComplement S H) (g : G) :
    (toLeftFun hS g : G)⁻¹ * g ∈ H :=
  QuotientGroup.leftRel_apply.mp <| Quotient.exact' <| quotientGroupMk_leftQuotientEquiv _ _


@[deprecated (since := "2024-12-28")]
alias _root_.Subgroup.MemLeftTransversals.inv_toFun_mul_mem := inv_toLeftFun_mul_mem


@[to_additive]
theorem inv_mul_toLeftFun_mem (hS : IsComplement S H) (g : G) :
    g⁻¹ * toLeftFun hS g ∈ H :=
                         /-
                           G : Type u_1
                           inst✝ : Group G
                           H : Subgroup G
                           S : Set G
                           hS : Subgroup.IsComplement S ↑H
                           g : G
                           ⊢ Eq (Inv.inv (HMul.hMul (Inv.inv ↑(hS.toLeftFun g)) g)) (HMul.hMul (Inv.inv g …
                         -/
  (congr_arg (· ∈ H) (by rw [mul_inv_rev, inv_inv])).mp (H.inv_mem (inv_toLeftFun_mul_mem hS g))
                         /-
                           🎉 no goals
                         -/


@[deprecated (since := "2024-12-28")]
alias _root_.Subgroup.MemLeftTransversals.inv_mul_toFun_mem := inv_mul_toLeftFun_mem


/-- A right transversal is in bijection with right cosets. -/
@[to_additive "A right transversal is in bijection with right cosets."]
noncomputable def rightQuotientEquiv (hT : IsComplement H T) :
    Quotient (QuotientGroup.rightRel H) ≃ T :=
  (Equiv.ofBijective _ (isComplement_subgroup_left_iff_bijective.mp hT)).symm


@[deprecated (since := "2024-12-28")]
alias _root_.Subgroup.MemRightTransversals.toEquiv := rightQuotientEquiv


/-- A right transversal is finite iff the subgroup has finite index. -/
@[to_additive "A right transversal is finite iff the subgroup has finite index."]
theorem finite_right_iff (h : IsComplement H T) : Finite T ↔ H.FiniteIndex := by
  rw [← h.rightQuotientEquiv.finite_iff,
    (QuotientGroup.quotientRightRelEquivQuotientLeftRel H).finite_iff]
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    T : Set G
    h : Subgroup.IsComplement (↑H) T
    ⊢ Iff (Finite (HasQuotient.Quotient G H)) H.FiniteIndex
  -/
  exact ⟨fun _ ↦ finiteIndex_of_finite_quotient H, fun _ ↦ finite_quotient_of_finiteIndex H⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-28")]
alias _root_.Subgroup.MemRightTransversals.finite_iff := finite_right_iff


@[to_additive]
theorem mk''_rightQuotientEquiv (hT : IsComplement H T)
     (q : Quotient (QuotientGroup.rightRel H)) : Quotient.mk'' (rightQuotientEquiv hT q : G) = q :=
  (rightQuotientEquiv hT).symm_apply_apply q


@[deprecated (since := "2024-12-28")]
alias _root_.Subgroup.MemRightTransversals.mk''_toEquiv := mk''_rightQuotientEquiv


@[to_additive]
theorem rightQuotientEquiv_apply {f : Quotient (QuotientGroup.rightRel H) → G}
    (hf : ∀ q, Quotient.mk'' (f q) = q) (q : Quotient (QuotientGroup.rightRel H)) :
    (rightQuotientEquiv (isComplement_range_right hf) q : G) = f q := by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    f : Quotient (QuotientGroup.rightRel H) → G
    hf : ∀ (q : Quotient (QuotientGroup.rightRel H)), Eq (Quotient.mk'' (f q)) q
    q : Quotient (QuotientGroup.rightRel H)
    ⊢ Eq (↑(⋯.rightQuotientEquiv q)) (f q)
  -/
  refine (Subtype.ext_iff.mp ?_).trans (Subtype.coe_mk (f q) ⟨q, rfl⟩)
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    f : Quotient (QuotientGroup.rightRel H) → G
    hf : ∀ (q : Quotient (QuotientGroup.rightRel H)), Eq (Quotient.mk'' (f q)) q
    q : Quotient (QuotientGroup.rightRel H)
    ⊢ Eq (⋯.rightQuotientEquiv q) ⟨f q, ⋯⟩
  -/
  exact (rightQuotientEquiv (isComplement_range_right hf)).apply_eq_iff_eq_symm_apply.2 (hf q).symm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-28")]
alias _root_.Subgroup.MemRightTransversals.toEquiv_apply := rightQuotientEquiv_apply


/-- A right transversal can be viewed as a function mapping each element of the group
  to the chosen representative from that right coset. -/
@[to_additive "A right transversal can be viewed as a function mapping each element of the group
  to the chosen representative from that right coset."]
noncomputable def toRightFun (hT : IsComplement H T) : G → T := rightQuotientEquiv hT ∘ .mk''


@[deprecated (since := "2024-12-28")]
alias _root_.Subgroup.MemRightTransversals.toFun := toRightFun


@[to_additive]
theorem mul_inv_toRightFun_mem (hT : IsComplement H T) (g : G) :
    g * (toRightFun hT g : G)⁻¹ ∈ H :=
  QuotientGroup.rightRel_apply.mp <| Quotient.exact' <| mk''_rightQuotientEquiv _ _


@[deprecated (since := "2024-12-28")]
alias _root_.Subgroup.MemRighTransversals.mul_inv_toFun_mem := mul_inv_toRightFun_mem


@[to_additive]
theorem toRightFun_mul_inv_mem (hT : IsComplement H T) (g : G) :
    (toRightFun hT g : G) * g⁻¹ ∈ H :=
                         /-
                           G : Type u_1
                           inst✝ : Group G
                           H : Subgroup G
                           T : Set G
                           hT : Subgroup.IsComplement (↑H) T
                           g : G
                           ⊢ Eq (Inv.inv (HMul.hMul g (Inv.inv ↑(hT.toRightFun g)))) (HMul.hMul (↑(hT.toR …
                         -/
  (congr_arg (· ∈ H) (by rw [mul_inv_rev, inv_inv])).mp (H.inv_mem (mul_inv_toRightFun_mem hT g))
                         /-
                           🎉 no goals
                         -/


@[deprecated (since := "2024-12-28")]
alias _root_.Subgroup.MemRighTransversals.toFun_mul_inv_mem := toRightFun_mul_inv_mem


/-- The collection of left transversals of a subgroup.-/
@[to_additive "The collection of left transversals of a subgroup."]
abbrev LeftTransversal (H : Subgroup G) := {S : Set G // IsComplement S H}


/-- The collection of right transversals of a subgroup.-/
@[to_additive "The collection of right transversals of a subgroup."]
abbrev RightTransversal (H : Subgroup G) := {T : Set G // IsComplement H T}


@[to_additive]
noncomputable instance : MulAction F H.LeftTransversal where
  smul f T :=
    ⟨f • (T : Set G), by
      /-
        G : Type u_1
        inst✝³ : Group G
        H K : Subgroup G
        S T✝ : Set G
        F : Type u_2
        inst✝² : Group F
        inst✝¹ : MulAction F G
        inst✝ : MulAction.QuotientAction F H
        f : F
        T : H.LeftTransversal
        ⊢ Subgroup.IsComplement (HSMul.hSMul f ↑T) ↑H
      -/
      refine isComplement_iff_existsUnique_inv_mul_mem.mpr fun g => ?_
      /-
        G : Type u_1
        inst✝³ : Group G
        H K : Subgroup G
        S T✝ : Set G
        F : Type u_2
        inst✝² : Group F
        inst✝¹ : MulAction F G
        inst✝ : MulAction.QuotientAction F H
        f : F
        T : H.LeftTransversal
        g : G
        ⊢ ExistsUnique fun s => Membership.mem (↑H) (HMul.hMul (Inv.inv ↑s) g)
      -/
      obtain ⟨t, ht1, ht2⟩ := isComplement_iff_existsUnique_inv_mul_mem.mp T.2 (f⁻¹ • g)
      /-
        case intro.intro
        G : Type u_1
        inst✝³ : Group G
        H K : Subgroup G
        S T✝ : Set G
        F : Type u_2
        inst✝² : Group F
        inst✝¹ : MulAction F G
        inst✝ : MulAction.QuotientAction F H
        f : F
        T : H.LeftTransversal
        g : G
        t : ↑↑T
        ht1 : Membership.mem (↑H) (HMul.hMul (Inv.inv ↑t) (HSMul.hSMul (Inv.inv f) g))
        ht2 : ∀ (y : ↑↑T), (fun s => Membership.mem (↑H) (HMul.hMul (Inv.inv ↑s) (HSMu …
        ⊢ ExistsUnique fun s => Membership.mem (↑H) (HMul.hMul (Inv.inv ↑s) g)
      -/
      refine ⟨⟨f • (t : G), Set.smul_mem_smul_set t.2⟩, ?_, ?_⟩
        /-
          case intro.intro.refine_1
          G : Type u_1
          inst✝³ : Group G
          H K : Subgroup G
          S T✝ : Set G
          F : Type u_2
          inst✝² : Group F
          inst✝¹ : MulAction F G
          inst✝ : MulAction.QuotientAction F H
          f : F
          T : H.LeftTransversal
          g : G
          t : ↑↑T
          ht1 : Membership.mem (↑H) (HMul.hMul (Inv.inv ↑t) (HSMul.hSMul (Inv.inv f) g))
          ht2 : ∀ (y : ↑↑T), (fun s => Membership.mem (↑H) (HMul.hMul (Inv.inv ↑s) (HSMu …
          ⊢ (fun s => Membership.mem (↑H) (HMul.hMul (Inv.inv ↑s) g)) ⟨HSMul.hSMul f ↑t, …
        -/
      · exact smul_inv_smul f g ▸ QuotientAction.inv_mul_mem f ht1
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.refine_2
          G : Type u_1
          inst✝³ : Group G
          H K : Subgroup G
          S T✝ : Set G
          F : Type u_2
          inst✝² : Group F
          inst✝¹ : MulAction F G
          inst✝ : MulAction.QuotientAction F H
          f : F
          T : H.LeftTransversal
          g : G
          t : ↑↑T
          ht1 : Membership.mem (↑H) (HMul.hMul (Inv.inv ↑t) (HSMul.hSMul (Inv.inv f) g))
          ht2 : ∀ (y : ↑↑T), (fun s => Membership.mem (↑H) (HMul.hMul (Inv.inv ↑s) (HSMu …
          ⊢ ∀ (y : ↑(HSMul.hSMul f ↑T)), (fun s => Membership.mem (↑H) (HMul.hMul (Inv.i …
        -/
      · rintro ⟨-, t', ht', rfl⟩ h
        /-
          case intro.intro.refine_2.mk.intro.intro
          G : Type u_1
          inst✝³ : Group G
          H K : Subgroup G
          S T✝ : Set G
          F : Type u_2
          inst✝² : Group F
          inst✝¹ : MulAction F G
          inst✝ : MulAction.QuotientAction F H
          f : F
          T : H.LeftTransversal
          g : G
          t : ↑↑T
          ht1 : Membership.mem (↑H) (HMul.hMul (Inv.inv ↑t) (HSMul.hSMul (Inv.inv f) g))
          ht2 : ∀ (y : ↑↑T), (fun s => Membership.mem (↑H) (HMul.hMul (Inv.inv ↑s) (HSMu …
          t' : G
          ht' : Membership.mem (↑T) t'
          h : Membership.mem (↑H) (HMul.hMul (Inv.inv ↑⟨(fun x => HSMul.hSMul f x) t', ⋯ …
          ⊢ Eq ⟨(fun x => HSMul.hSMul f x) t', ⋯⟩ ⟨HSMul.hSMul f ↑t, ⋯⟩
        -/
        replace h := QuotientAction.inv_mul_mem f⁻¹ h
        /-
          case intro.intro.refine_2.mk.intro.intro
          G : Type u_1
          inst✝³ : Group G
          H K : Subgroup G
          S T✝ : Set G
          F : Type u_2
          inst✝² : Group F
          inst✝¹ : MulAction F G
          inst✝ : MulAction.QuotientAction F H
          f : F
          T : H.LeftTransversal
          g : G
          t : ↑↑T
          ht1 : Membership.mem (↑H) (HMul.hMul (Inv.inv ↑t) (HSMul.hSMul (Inv.inv f) g))
          ht2 : ∀ (y : ↑↑T), (fun s => Membership.mem (↑H) (HMul.hMul (Inv.inv ↑s) (HSMu …
          t' : G
          ht' : Membership.mem (↑T) t'
          h : Membership.mem H (HMul.hMul (Inv.inv (HSMul.hSMul (Inv.inv f) ↑⟨(fun x =>  …
          ⊢ Eq ⟨(fun x => HSMul.hSMul f x) t', ⋯⟩ ⟨HSMul.hSMul f ↑t, ⋯⟩
        -/
        simp only [Subtype.ext_iff, Subtype.coe_mk, smul_left_cancel_iff, inv_smul_smul] at h ⊢
        /-
          case intro.intro.refine_2.mk.intro.intro
          G : Type u_1
          inst✝³ : Group G
          H K : Subgroup G
          S T✝ : Set G
          F : Type u_2
          inst✝² : Group F
          inst✝¹ : MulAction F G
          inst✝ : MulAction.QuotientAction F H
          f : F
          T : H.LeftTransversal
          g : G
          t : ↑↑T
          ht1 : Membership.mem (↑H) (HMul.hMul (Inv.inv ↑t) (HSMul.hSMul (Inv.inv f) g))
          ht2 : ∀ (y : ↑↑T), (fun s => Membership.mem (↑H) (HMul.hMul (Inv.inv ↑s) (HSMu …
          t' : G
          ht' : Membership.mem (↑T) t'
          h : Membership.mem H (HMul.hMul (Inv.inv t') (HSMul.hSMul (Inv.inv f) g))
          ⊢ Eq t' ↑t
        -/
        exact Subtype.ext_iff.mp (ht2 ⟨t', ht'⟩ h)⟩
        /-
          🎉 no goals
        -/
  one_smul T := Subtype.ext (one_smul F (T : Set G))
  mul_smul f₁ f₂ T := Subtype.ext (mul_smul f₁ f₂ (T : Set G))


@[to_additive]
theorem smul_toLeftFun (f : F) (S : H.LeftTransversal) (g : G) :
    (f • (S.2.toLeftFun g : G)) = (f • S).2.toLeftFun (f • g) :=
  Subtype.ext_iff.mp <| @ExistsUnique.unique (↥(f • (S : Set G))) (fun s => (↑s)⁻¹ * f • g ∈ H)
    (isComplement_iff_existsUnique_inv_mul_mem.mp (f • S).2 (f • g))
    ⟨f • (S.2.toLeftFun g : G), Set.smul_mem_smul_set (Subtype.coe_prop _)⟩
      ((f • S).2.toLeftFun (f • g))
    (QuotientAction.inv_mul_mem f (S.2.inv_toLeftFun_mul_mem g))
      ((f • S).2.inv_toLeftFun_mul_mem (f • g))


@[to_additive]
theorem smul_leftQuotientEquiv (f : F) (S : H.LeftTransversal) (q : G ⧸ H) :
    f • (S.2.leftQuotientEquiv  q : G) = (f • S).2.leftQuotientEquiv (f • q) :=
  Quotient.inductionOn' q fun g => smul_toLeftFun f S g


@[to_additive]
theorem smul_apply_eq_smul_apply_inv_smul (f : F) (S : H.LeftTransversal) (q : G ⧸ H) :
    ((f • S).2.leftQuotientEquiv q : G) = f • (S.2.leftQuotientEquiv (f⁻¹ • q) : G) := by
  /-
    G : Type u_1
    inst✝³ : Group G
    H : Subgroup G
    F : Type u_2
    inst✝² : Group F
    inst✝¹ : MulAction F G
    inst✝ : MulAction.QuotientAction F H
    f : F
    S : H.LeftTransversal
    q : HasQuotient.Quotient G H
    ⊢ Eq (↑(⋯.leftQuotientEquiv q)) (HSMul.hSMul f ↑(⋯.leftQuotientEquiv (HSMul.hS …
  -/
  rw [smul_leftQuotientEquiv, smul_inv_smul]
  /-
    🎉 no goals
  -/


@[to_additive]
instance : Inhabited H.LeftTransversal :=
  ⟨⟨Set.range Quotient.out, isComplement_range_left Quotient.out_eq'⟩⟩


@[to_additive]
instance : Inhabited H.RightTransversal :=
  ⟨⟨Set.range Quotient.out, isComplement_range_right Quotient.out_eq'⟩⟩


theorem IsComplement'.isCompl (h : IsComplement' H K) : IsCompl H K := by
  refine
    ⟨disjoint_iff_inf_le.mpr fun g ⟨p, q⟩ =>
        let x : H × K := ⟨⟨g, p⟩, 1⟩
        let y : H × K := ⟨1, g, q⟩
        Subtype.ext_iff.mp
          (Prod.ext_iff.mp (show x = y from h.1 ((mul_one g).trans (one_mul g).symm))).1,
      codisjoint_iff_le_sup.mpr fun g _ => ?_⟩
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    h : H.IsComplement' K
    g : G
    x✝ : Membership.mem Top.top g
    ⊢ Membership.mem (Max.max H K) g
  -/
  obtain ⟨⟨h, k⟩, rfl⟩ := h.2 g
  /-
    case intro.mk
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    h✝ : H.IsComplement' K
    h : ↑↑H
    k : ↑↑K
    x✝ : Membership.mem Top.top ((fun x => HMul.hMul ↑x.1 ↑x.2) { fst := h, snd := …
    ⊢ Membership.mem (Max.max H K) ((fun x => HMul.hMul ↑x.1 ↑x.2) { fst := h, snd …
  -/
  exact Subgroup.mul_mem_sup h.2 k.2
  /-
    🎉 no goals
  -/


theorem IsComplement'.sup_eq_top (h : IsComplement' H K) : H ⊔ K = ⊤ :=
  h.isCompl.sup_eq_top


theorem IsComplement'.disjoint (h : IsComplement' H K) : Disjoint H K :=
  h.isCompl.disjoint


theorem IsComplement'.index_eq_card (h : IsComplement' H K) : K.index = Nat.card H :=
  h.card_left.symm


/-- If `H` and `K` are complementary with `K` normal, then `G ⧸ K` is isomorphic to `H`. -/
@[simps!]
noncomputable def IsComplement'.QuotientMulEquiv [K.Normal] (h : H.IsComplement' K) :
    G ⧸ K ≃* H :=
  MulEquiv.symm
  { h.leftQuotientEquiv.symm with
    map_mul' := fun _ _ ↦ rfl }


theorem IsComplement.card_mul (h : IsComplement S T) :
    Nat.card S * Nat.card T = Nat.card G :=
  (Nat.card_prod _ _).symm.trans (Nat.card_eq_of_bijective _ h)


theorem IsComplement'.card_mul (h : IsComplement' H K) :
    Nat.card H * Nat.card K = Nat.card G :=
  IsComplement.card_mul h


theorem isComplement'_of_disjoint_and_mul_eq_univ (h1 : Disjoint H K)
    (h2 : ↑H * ↑K = (Set.univ : Set G)) : IsComplement' H K := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    h1 : Disjoint H K
    h2 : Eq (HMul.hMul ↑H ↑K) Set.univ
    ⊢ H.IsComplement' K
  -/
  refine ⟨mul_injective_of_disjoint h1, fun g => ?_⟩
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    h1 : Disjoint H K
    h2 : Eq (HMul.hMul ↑H ↑K) Set.univ
    g : G
    ⊢ Exists fun a => Eq ((fun x => HMul.hMul ↑x.1 ↑x.2) a) g
  -/
  obtain ⟨h, hh, k, hk, hg⟩ := Set.eq_univ_iff_forall.mp h2 g
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    h1 : Disjoint H K
    h2 : Eq (HMul.hMul ↑H ↑K) Set.univ
    g h : G
    hh : Membership.mem (↑H) h
    k : G
    hk : Membership.mem (↑K) k
    hg : Eq ((fun x1 x2 => HMul.hMul x1 x2) h k) g
    ⊢ Exists fun a => Eq ((fun x => HMul.hMul ↑x.1 ↑x.2) a) g
  -/
  exact ⟨(⟨h, hh⟩, ⟨k, hk⟩), hg⟩
  /-
    🎉 no goals
  -/


theorem isComplement'_of_card_mul_and_disjoint [Finite G]
    (h1 : Nat.card H * Nat.card K = Nat.card G) (h2 : Disjoint H K) :
    IsComplement' H K :=
  (Nat.bijective_iff_injective_and_card _).mpr
    ⟨mul_injective_of_disjoint h2, (Nat.card_prod H K).trans h1⟩


theorem isComplement'_iff_card_mul_and_disjoint [Finite G] :
    IsComplement' H K ↔ Nat.card H * Nat.card K = Nat.card G ∧ Disjoint H K :=
  ⟨fun h => ⟨h.card_mul, h.disjoint⟩, fun h => isComplement'_of_card_mul_and_disjoint h.1 h.2⟩


theorem isComplement'_of_coprime [Finite G]
    (h1 : Nat.card H * Nat.card K = Nat.card G)
    (h2 : Nat.Coprime (Nat.card H) (Nat.card K)) : IsComplement' H K :=
  isComplement'_of_card_mul_and_disjoint h1 (disjoint_iff.mpr (inf_eq_bot_of_coprime h2))


theorem isComplement'_stabilizer {α : Type*} [MulAction G α] (a : α)
    (h1 : ∀ h : H, h • a = a → h = 1) (h2 : ∀ g : G, ∃ h : H, h • g • a = a) :
    IsComplement' H (MulAction.stabilizer G a) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    α : Type u_2
    inst✝ : MulAction G α
    a : α
    h1 : ∀ (h : Subtype fun x => Membership.mem H x), Eq (HSMul.hSMul h a) a → Eq  …
    h2 : ∀ (g : G), Exists fun h => Eq (HSMul.hSMul h (HSMul.hSMul g a)) a
    ⊢ H.IsComplement' (MulAction.stabilizer G a)
  -/
  refine isComplement_iff_existsUnique.mpr fun g => ?_
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    α : Type u_2
    inst✝ : MulAction G α
    a : α
    h1 : ∀ (h : Subtype fun x => Membership.mem H x), Eq (HSMul.hSMul h a) a → Eq  …
    h2 : ∀ (g : G), Exists fun h => Eq (HSMul.hSMul h (HSMul.hSMul g a)) a
    g : G
    ⊢ ExistsUnique fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g
  -/
  obtain ⟨h, hh⟩ := h2 g
  /-
    case intro
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    α : Type u_2
    inst✝ : MulAction G α
    a : α
    h1 : ∀ (h : Subtype fun x => Membership.mem H x), Eq (HSMul.hSMul h a) a → Eq  …
    h2 : ∀ (g : G), Exists fun h => Eq (HSMul.hSMul h (HSMul.hSMul g a)) a
    g : G
    h : Subtype fun x => Membership.mem H x
    hh : Eq (HSMul.hSMul h (HSMul.hSMul g a)) a
    ⊢ ExistsUnique fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g
  -/
  have hh' : (↑h * g) • a = a := by rwa [mul_smul]
  /-
    case intro
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    α : Type u_2
    inst✝ : MulAction G α
    a : α
    h1 : ∀ (h : Subtype fun x => Membership.mem H x), Eq (HSMul.hSMul h a) a → Eq  …
    h2 : ∀ (g : G), Exists fun h => Eq (HSMul.hSMul h (HSMul.hSMul g a)) a
    g : G
    h : Subtype fun x => Membership.mem H x
    hh : Eq (HSMul.hSMul h (HSMul.hSMul g a)) a
    hh' : Eq (HSMul.hSMul (HMul.hMul (↑h) g) a) a
    ⊢ ExistsUnique fun x => Eq (HMul.hMul ↑x.1 ↑x.2) g
  -/
  refine ⟨⟨h⁻¹, h * g, hh'⟩, inv_mul_cancel_left ↑h g, ?_⟩
  /-
    case intro
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    α : Type u_2
    inst✝ : MulAction G α
    a : α
    h1 : ∀ (h : Subtype fun x => Membership.mem H x), Eq (HSMul.hSMul h a) a → Eq  …
    h2 : ∀ (g : G), Exists fun h => Eq (HSMul.hSMul h (HSMul.hSMul g a)) a
    g : G
    h : Subtype fun x => Membership.mem H x
    hh : Eq (HSMul.hSMul h (HSMul.hSMul g a)) a
    hh' : Eq (HSMul.hSMul (HMul.hMul (↑h) g) a) a
    ⊢ ∀ (y : Prod ↑↑H ↑↑(MulAction.stabilizer G a)), (fun x => Eq (HMul.hMul ↑x.1  …
  -/
  rintro ⟨h', g, hg : g • a = a⟩ rfl
  /-
    case intro.mk.mk
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    α : Type u_2
    inst✝ : MulAction G α
    a : α
    h1 : ∀ (h : Subtype fun x => Membership.mem H x), Eq (HSMul.hSMul h a) a → Eq  …
    h2 : ∀ (g : G), Exists fun h => Eq (HSMul.hSMul h (HSMul.hSMul g a)) a
    h : Subtype fun x => Membership.mem H x
    h' : ↑↑H
    g : G
    hg : Eq (HSMul.hSMul g a) a
    hh : Eq (HSMul.hSMul h (HSMul.hSMul (HMul.hMul ↑{ fst := h', snd := ⟨g, hg⟩ }. …
    hh' : Eq (HSMul.hSMul (HMul.hMul (↑h) (HMul.hMul ↑{ fst := h', snd := ⟨g, hg⟩  …
    ⊢ Eq { fst := h', snd := ⟨g, hg⟩ } { fst := Inv.inv h, snd := ⟨HMul.hMul (↑h)  …
  -/
  specialize h1 (h * h') (by rwa [mul_smul, smul_def h', ← hg, ← mul_smul, hg])
  /-
    case intro.mk.mk
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    α : Type u_2
    inst✝ : MulAction G α
    a : α
    h2 : ∀ (g : G), Exists fun h => Eq (HSMul.hSMul h (HSMul.hSMul g a)) a
    h : Subtype fun x => Membership.mem H x
    h' : ↑↑H
    g : G
    hg : Eq (HSMul.hSMul g a) a
    hh : Eq (HSMul.hSMul h (HSMul.hSMul (HMul.hMul ↑{ fst := h', snd := ⟨g, hg⟩ }. …
    hh' : Eq (HSMul.hSMul (HMul.hMul (↑h) (HMul.hMul ↑{ fst := h', snd := ⟨g, hg⟩  …
    h1 : Eq (HMul.hMul h h') 1
    ⊢ Eq { fst := h', snd := ⟨g, hg⟩ } { fst := Inv.inv h, snd := ⟨HMul.hMul (↑h)  …
  -/
  refine Prod.ext (eq_inv_of_mul_eq_one_right h1) (Subtype.ext ?_)
  /-
    case intro.mk.mk
    G : Type u_1
    inst✝¹ : Group G
    H : Subgroup G
    α : Type u_2
    inst✝ : MulAction G α
    a : α
    h2 : ∀ (g : G), Exists fun h => Eq (HSMul.hSMul h (HSMul.hSMul g a)) a
    h : Subtype fun x => Membership.mem H x
    h' : ↑↑H
    g : G
    hg : Eq (HSMul.hSMul g a) a
    hh : Eq (HSMul.hSMul h (HSMul.hSMul (HMul.hMul ↑{ fst := h', snd := ⟨g, hg⟩ }. …
    hh' : Eq (HSMul.hSMul (HMul.hMul (↑h) (HMul.hMul ↑{ fst := h', snd := ⟨g, hg⟩  …
    h1 : Eq (HMul.hMul h h') 1
    ⊢ Eq ↑{ fst := h', snd := ⟨g, hg⟩ }.2 ↑{ fst := Inv.inv h, snd := ⟨HMul.hMul ( …
  -/
  rwa [Subtype.ext_iff, coe_one, coe_mul, ← self_eq_mul_left, mul_assoc (↑h) (↑h') g] at h1
  /-
    🎉 no goals
  -/



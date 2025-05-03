/-- A predicate on a monoid saying that all elements are of finite order. -/
@[to_additive "A predicate on an additive monoid saying that all elements are of finite order."]
def IsTorsion :=
  ∀ g : G, IsOfFinOrder g


/-- A monoid is not a torsion monoid if it has an element of infinite order. -/
@[to_additive (attr := simp) "An additive monoid is not a torsion monoid if it
  has an element of infinite order."]
theorem not_isTorsion_iff : ¬IsTorsion G ↔ ∃ g : G, ¬IsOfFinOrder g := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    ⊢ Iff (Not (Monoid.IsTorsion G)) (Exists fun g => Not (IsOfFinOrder g))
  -/
  rw [IsTorsion, not_forall]
  /-
    🎉 no goals
  -/


/-- Torsion monoids are really groups. -/
@[to_additive "Torsion additive monoids are really additive groups"]
noncomputable def IsTorsion.group [Monoid G] (tG : IsTorsion G) : Group G :=
  { ‹Monoid G› with
    inv := fun g => g ^ (orderOf g - 1)
    inv_mul_cancel := fun g => by
      /-
        G : Type u_1
        H : Type u_2
        inst✝ : Monoid G
        tG : Monoid.IsTorsion G
        g : G
        ⊢ Eq (HMul.hMul (Inv.inv g) g) 1
      -/
      erw [← pow_succ, tsub_add_cancel_of_le, pow_orderOf_eq_one]
      /-
        G : Type u_1
        H : Type u_2
        inst✝ : Monoid G
        tG : Monoid.IsTorsion G
        g : G
        ⊢ LE.le 1 (orderOf g)
      -/
      exact (tG g).orderOf_pos }
      /-
        🎉 no goals
      -/


/-- Subgroups of torsion groups are torsion groups. -/
@[to_additive "Subgroups of additive torsion groups are additive torsion groups."]
theorem IsTorsion.subgroup (tG : IsTorsion G) (H : Subgroup G) : IsTorsion H := fun h =>
  Submonoid.isOfFinOrder_coe.1 <| tG h


/-- The image of a surjective torsion group homomorphism is torsion. -/
@[to_additive AddIsTorsion.of_surjective
      "The image of a surjective additive torsion group homomorphism is torsion."]
theorem IsTorsion.of_surjective {f : G →* H} (hf : Function.Surjective f) (tG : IsTorsion G) :
    IsTorsion H := fun h => by
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    inst✝ : Group H
    f : MonoidHom G H
    hf : Function.Surjective ⇑f
    tG : Monoid.IsTorsion G
    h : H
    ⊢ IsOfFinOrder h
  -/
  obtain ⟨g, hg⟩ := hf h
  /-
    case intro
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    inst✝ : Group H
    f : MonoidHom G H
    hf : Function.Surjective ⇑f
    tG : Monoid.IsTorsion G
    h : H
    g : G
    hg : Eq (f g) h
    ⊢ IsOfFinOrder h
  -/
  rw [← hg]
  /-
    case intro
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    inst✝ : Group H
    f : MonoidHom G H
    hf : Function.Surjective ⇑f
    tG : Monoid.IsTorsion G
    h : H
    g : G
    hg : Eq (f g) h
    ⊢ IsOfFinOrder (f g)
  -/
  exact f.isOfFinOrder (tG g)
  /-
    🎉 no goals
  -/


/-- Torsion groups are closed under extensions. -/
@[to_additive AddIsTorsion.extension_closed "Additive torsion groups are closed under extensions."]
theorem IsTorsion.extension_closed {f : G →* H} (hN : N = f.ker) (tH : IsTorsion H)
    (tN : IsTorsion N) : IsTorsion G := fun g => by
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    N : Subgroup G
    inst✝ : Group H
    f : MonoidHom G H
    hN : Eq N f.ker
    tH : Monoid.IsTorsion H
    tN : Monoid.IsTorsion (Subtype fun x => Membership.mem N x)
    g : G
    ⊢ IsOfFinOrder g
  -/
  obtain ⟨ngn, ngnpos, hngn⟩ := (tH <| f g).exists_pow_eq_one
  /-
    case intro.intro
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    N : Subgroup G
    inst✝ : Group H
    f : MonoidHom G H
    hN : Eq N f.ker
    tH : Monoid.IsTorsion H
    tN : Monoid.IsTorsion (Subtype fun x => Membership.mem N x)
    g : G
    ngn : Nat
    ngnpos : LT.lt 0 ngn
    hngn : Eq (HPow.hPow (f g) ngn) 1
    ⊢ IsOfFinOrder g
  -/
  have hmem := MonoidHom.mem_ker.mpr ((f.map_pow g ngn).trans hngn)
  /-
    case intro.intro
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    N : Subgroup G
    inst✝ : Group H
    f : MonoidHom G H
    hN : Eq N f.ker
    tH : Monoid.IsTorsion H
    tN : Monoid.IsTorsion (Subtype fun x => Membership.mem N x)
    g : G
    ngn : Nat
    ngnpos : LT.lt 0 ngn
    hngn : Eq (HPow.hPow (f g) ngn) 1
    hmem : Membership.mem f.ker (HPow.hPow g ngn)
    ⊢ IsOfFinOrder g
  -/
  lift g ^ ngn to N using hN.symm ▸ hmem with gn h
  /-
    case intro.intro.intro
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    N : Subgroup G
    inst✝ : Group H
    f : MonoidHom G H
    hN : Eq N f.ker
    tH : Monoid.IsTorsion H
    tN : Monoid.IsTorsion (Subtype fun x => Membership.mem N x)
    g : G
    ngn : Nat
    ngnpos : LT.lt 0 ngn
    hngn : Eq (HPow.hPow (f g) ngn) 1
    gn : Subtype fun x => Membership.mem N x
    h : Eq (↑gn) (HPow.hPow g ngn)
    hmem✝ hmem : Membership.mem f.ker ↑gn
    ⊢ IsOfFinOrder g
  -/
  obtain ⟨nn, nnpos, hnn⟩ := (tN gn).exists_pow_eq_one
  exact isOfFinOrder_iff_pow_eq_one.mpr <| ⟨ngn * nn, mul_pos ngnpos nnpos, by
      rw [pow_mul, ← h, ← Subgroup.coe_pow, hnn, Subgroup.coe_one]⟩


/-- The image of a quotient is torsion iff the group is torsion. -/
@[to_additive AddIsTorsion.quotient_iff
      "The image of a quotient is additively torsion iff the group is torsion."]
theorem IsTorsion.quotient_iff {f : G →* H} (hf : Function.Surjective f) (hN : N = f.ker)
    (tN : IsTorsion N) : IsTorsion H ↔ IsTorsion G :=
  ⟨fun tH => IsTorsion.extension_closed hN tH tN, fun tG => IsTorsion.of_surjective hf tG⟩


/-- If a group exponent exists, the group is torsion. -/
@[to_additive ExponentExists.is_add_torsion
      "If a group exponent exists, the group is additively torsion."]
theorem ExponentExists.isTorsion (h : ExponentExists G) : IsTorsion G := fun g => by
  /-
    G : Type u_1
    inst✝ : Group G
    h : Monoid.ExponentExists G
    g : G
    ⊢ IsOfFinOrder g
  -/
  obtain ⟨n, npos, hn⟩ := h
  /-
    case intro.intro
    G : Type u_1
    inst✝ : Group G
    g : G
    n : Nat
    npos : LT.lt 0 n
    hn : ∀ (g : G), Eq (HPow.hPow g n) 1
    ⊢ IsOfFinOrder g
  -/
  exact isOfFinOrder_iff_pow_eq_one.mpr ⟨n, npos, hn g⟩
  /-
    🎉 no goals
  -/


/-- The group exponent exists for any bounded torsion group. -/
@[to_additive IsAddTorsion.exponentExists
      "The group exponent exists for any bounded additive torsion group."]
theorem IsTorsion.exponentExists (tG : IsTorsion G)
    (bounded : (Set.range fun g : G => orderOf g).Finite) : ExponentExists G :=
  exponent_ne_zero.mp <|
    (exponent_ne_zero_iff_range_orderOf_finite fun g => (tG g).orderOf_pos).mpr bounded


/-- Finite groups are torsion groups. -/
@[to_additive is_add_torsion_of_finite "Finite additive groups are additive torsion groups."]
theorem isTorsion_of_finite [Finite G] : IsTorsion G :=
  ExponentExists.isTorsion .of_finite


/-- A module whose scalars are additively torsion is additively torsion. -/
theorem IsTorsion.module_of_torsion [Semiring R] [Module R M] (tR : IsTorsion R) : IsTorsion M :=
  fun f =>
  isOfFinAddOrder_iff_nsmul_eq_zero.mpr <| by
    /-
      R : Type u_3
      M : Type u_4
      inst✝² : AddCommMonoid M
      inst✝¹ : Semiring R
      inst✝ : Module R M
      tR : AddMonoid.IsTorsion R
      f : M
      ⊢ Exists fun n => And (LT.lt 0 n) (Eq (HSMul.hSMul n f) 0)
    -/
    obtain ⟨n, npos, hn⟩ := (tR 1).exists_nsmul_eq_zero
    /-
      case intro.intro
      R : Type u_3
      M : Type u_4
      inst✝² : AddCommMonoid M
      inst✝¹ : Semiring R
      inst✝ : Module R M
      tR : AddMonoid.IsTorsion R
      f : M
      n : Nat
      npos : LT.lt 0 n
      hn : Eq (HSMul.hSMul n 1) 0
      ⊢ Exists fun n => And (LT.lt 0 n) (Eq (HSMul.hSMul n f) 0)
    -/
    exact ⟨n, npos, by simp only [← Nat.cast_smul_eq_nsmul R _ f, ← nsmul_one, hn, zero_smul]⟩
    /-
      🎉 no goals
    -/


/-- A module with a finite ring of scalars is additively torsion. -/
theorem IsTorsion.module_of_finite [Ring R] [Finite R] [Module R M] : IsTorsion M :=
  (is_add_torsion_of_finite : IsTorsion R).module_of_torsion _ _


/-- The torsion submonoid of a commutative monoid.

(Note that by `Monoid.IsTorsion.group` torsion monoids are truthfully groups.)
-/
@[to_additive addTorsion "The torsion submonoid of an additive commutative monoid."]
def torsion : Submonoid G where
  carrier := { x | IsOfFinOrder x }
  one_mem' := IsOfFinOrder.one
  mul_mem' hx hy := hx.mul hy


/-- Torsion submonoids are torsion. -/
@[to_additive "Additive torsion submonoids are additively torsion."]
theorem torsion.isTorsion : IsTorsion <| torsion G := fun ⟨x, n, npos, hn⟩ =>
  ⟨n, npos,
    Subtype.ext <| by
      /-
        G : Type u_1
        inst✝ : CommMonoid G
        x✝ : Subtype fun x => Membership.mem (CommMonoid.torsion G) x
        x : G
        n : Nat
        npos : GT.gt n 0
        hn : Function.IsPeriodicPt (fun x_1 => HMul.hMul x x_1) n 1
        ⊢ Eq ↑(Nat.iterate (fun x_1 => HMul.hMul ⟨x, ⋯⟩ x_1) n 1) ↑1
      -/
      dsimp
      /-
        G : Type u_1
        inst✝ : CommMonoid G
        x✝ : Subtype fun x => Membership.mem (CommMonoid.torsion G) x
        x : G
        n : Nat
        npos : GT.gt n 0
        hn : Function.IsPeriodicPt (fun x_1 => HMul.hMul x x_1) n 1
        ⊢ Eq (↑(Nat.iterate (fun x_1 => HMul.hMul ⟨x, ⋯⟩ x_1) n 1)) 1
      -/
      rw [mul_left_iterate]
      /-
        G : Type u_1
        inst✝ : CommMonoid G
        x✝ : Subtype fun x => Membership.mem (CommMonoid.torsion G) x
        x : G
        n : Nat
        npos : GT.gt n 0
        hn : Function.IsPeriodicPt (fun x_1 => HMul.hMul x x_1) n 1
        ⊢ Eq (↑((fun x_1 => HMul.hMul (HPow.hPow ⟨x, ⋯⟩ n) x_1) 1)) 1
      -/
      change _ * 1 = 1
      rw [_root_.mul_one, SubmonoidClass.coe_pow, Subtype.coe_mk,
        (isPeriodicPt_mul_iff_pow_eq_one _).mp hn]⟩


/-- The `p`-primary component is the submonoid of elements with order prime-power of `p`. -/
@[to_additive (attr := simps)
      "The `p`-primary component is the submonoid of elements with additive
      order prime-power of `p`."]
def primaryComponent : Submonoid G where
  carrier := { g | ∃ n : ℕ, orderOf g = p ^ n }
                     /-
                       G : Type u_1
                       H : Type u_2
                       inst✝ : CommMonoid G
                       p : Nat
                       hp : Fact (Nat.Prime p)
                       ⊢ Eq (orderOf 1) (HPow.hPow p 0)
                     -/
  one_mem' := ⟨0, by rw [pow_zero, orderOf_one]⟩
                     /-
                       🎉 no goals
                     -/
      /-
        G : Type u_1
        H : Type u_2
        inst✝ : CommMonoid G
        p : Nat
        hp : Fact (Nat.Prime p)
        a✝ b✝ : G
        hg₁ : Membership.mem (setOf fun g => Exists fun n => Eq (orderOf g) (HPow.hPow …
        hg₂ : Membership.mem (setOf fun g => Exists fun n => Eq (orderOf g) (HPow.hPow …
        ⊢ Exists fun m => Eq (HPow.hPow (HMul.hMul a✝ b✝) (HPow.hPow p m)) 1
      -/
  mul_mem' hg₁ hg₂ :=
      /-
        case intro
        G : Type u_1
        H : Type u_2
        inst✝ : CommMonoid G
        p : Nat
        hp : Fact (Nat.Prime p)
        a✝ b✝ : G
        hg₁ : Membership.mem (setOf fun g => Exists fun n => Eq (orderOf g) (HPow.hPow …
        hg₂ : Membership.mem (setOf fun g => Exists fun n => Eq (orderOf g) (HPow.hPow …
        m : Nat
        hm : Eq (HPow.hPow a✝ (HPow.hPow p m)) 1
        ⊢ Exists fun m => Eq (HPow.hPow (HMul.hMul a✝ b✝) (HPow.hPow p m)) 1
      -/
    exists_orderOf_eq_prime_pow_iff.mpr <| by
      obtain ⟨m, hm⟩ := exists_orderOf_eq_prime_pow_iff.mp hg₁
      obtain ⟨n, hn⟩ := exists_orderOf_eq_prime_pow_iff.mp hg₂
      exact
        ⟨m + n, by
          rw [mul_pow, pow_add, pow_mul, hm, one_pow, Monoid.one_mul, mul_comm, pow_mul, hn,
            one_pow]⟩


/-- Elements of the `p`-primary component have order `p^n` for some `n`. -/
@[to_additive primaryComponent.exists_orderOf_eq_prime_nsmul
  "Elements of the `p`-primary component have additive order `p^n` for some `n`"]
theorem primaryComponent.exists_orderOf_eq_prime_pow (g : CommMonoid.primaryComponent G p) :
    ∃ n : ℕ, orderOf g = p ^ n := by
      /-
        G : Type u_1
        inst✝ : CommMonoid G
        p : Nat
        hp : Fact (Nat.Prime p)
        g : Subtype fun x => Membership.mem (CommMonoid.primaryComponent G p) x
        ⊢ Exists fun n => Eq (orderOf g) (HPow.hPow p n)
      -/
      obtain ⟨_, hn⟩ := g.property
      /-
        case intro
        G : Type u_1
        inst✝ : CommMonoid G
        p : Nat
        hp : Fact (Nat.Prime p)
        g : Subtype fun x => Membership.mem (CommMonoid.primaryComponent G p) x
        w✝ : Nat
        hn : Eq (orderOf ↑g) (HPow.hPow p w✝)
        ⊢ Exists fun n => Eq (orderOf g) (HPow.hPow p n)
      -/
      rw [orderOf_submonoid g] at hn
      /-
        case intro
        G : Type u_1
        inst✝ : CommMonoid G
        p : Nat
        hp : Fact (Nat.Prime p)
        g : Subtype fun x => Membership.mem (CommMonoid.primaryComponent G p) x
        w✝ : Nat
        hn : Eq (orderOf g) (HPow.hPow p w✝)
        ⊢ Exists fun n => Eq (orderOf g) (HPow.hPow p n)
      -/
      exact ⟨_, hn⟩
      /-
        🎉 no goals
      -/


/-- The `p`- and `q`-primary components are disjoint for `p ≠ q`. -/
@[to_additive "The `p`- and `q`-primary components are disjoint for `p ≠ q`."]
theorem primaryComponent.disjoint {p' : ℕ} [hp' : Fact p'.Prime] (hne : p ≠ p') :
    Disjoint (CommMonoid.primaryComponent G p) (CommMonoid.primaryComponent G p') :=
  Submonoid.disjoint_def.mpr <| by
    /-
      G : Type u_1
      inst✝ : CommMonoid G
      p : Nat
      hp : Fact (Nat.Prime p)
      p' : Nat
      hp' : Fact (Nat.Prime p')
      hne : Ne p p'
      ⊢ ∀ {x : G}, Membership.mem (CommMonoid.primaryComponent G p) x → Membership.m …
    -/
    rintro g ⟨_ | n, hn⟩ ⟨n', hn'⟩
      /-
        case intro.zero.intro
        G : Type u_1
        inst✝ : CommMonoid G
        p : Nat
        hp : Fact (Nat.Prime p)
        p' : Nat
        hp' : Fact (Nat.Prime p')
        hne : Ne p p'
        g : G
        hn : Eq (orderOf g) (HPow.hPow p 0)
        n' : Nat
        hn' : Eq (orderOf g) (HPow.hPow p' n')
        ⊢ Eq g 1
      -/
    · rwa [pow_zero, orderOf_eq_one_iff] at hn
      /-
        🎉 no goals
      -/
    · exact
        absurd (eq_of_prime_pow_eq hp.out.prime hp'.out.prime n.succ_pos (hn.symm.trans hn')) hne


/-- The torsion submonoid of a torsion monoid is `⊤`. -/
@[to_additive (attr := simp) "The additive torsion submonoid of an additive torsion monoid is `⊤`."]
                                                                /-
                                                                  G : Type u_1
                                                                  inst✝ : CommMonoid G
                                                                  tG : Monoid.IsTorsion G
                                                                  ⊢ Eq (CommMonoid.torsion G) Top.top
                                                                -/
theorem torsion_eq_top (tG : IsTorsion G) : torsion G = ⊤ := by ext; tauto
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


/-- A torsion monoid is isomorphic to its torsion submonoid. -/
@[to_additive "An additive torsion monoid is isomorphic to its torsion submonoid."]
def torsionMulEquiv (tG : IsTorsion G) : torsion G ≃* G :=
  (MulEquiv.submonoidCongr tG.torsion_eq_top).trans Submonoid.topEquiv


@[to_additive]
theorem torsionMulEquiv_apply (tG : IsTorsion G) (a : torsion G) :
    tG.torsionMulEquiv a = MulEquiv.submonoidCongr tG.torsion_eq_top a :=
  rfl


@[to_additive]
theorem torsionMulEquiv_symm_apply_coe (tG : IsTorsion G) (a : G) :
    tG.torsionMulEquiv.symm a = ⟨Submonoid.topEquiv.symm a, tG _⟩ :=
  rfl


/-- Torsion submonoids of a torsion submonoid are isomorphic to the submonoid. -/
@[to_additive (attr := simp) AddCommMonoid.Torsion.ofTorsion
      "Additive torsion submonoids of an additive torsion submonoid are
      isomorphic to the submonoid."]
def Torsion.ofTorsion : torsion (torsion G) ≃* torsion G :=
  Monoid.IsTorsion.torsionMulEquiv CommMonoid.torsion.isTorsion


/-- The torsion subgroup of an abelian group. -/
@[to_additive "The torsion subgroup of an additive abelian group."]
def torsion : Subgroup G :=
  { CommMonoid.torsion G with inv_mem' := fun hx => IsOfFinOrder.inv hx }


/-- The torsion submonoid of an abelian group equals the torsion subgroup as a submonoid. -/
@[to_additive add_torsion_eq_add_torsion_submonoid
      "The additive torsion submonoid of an abelian group equals the torsion
      subgroup as a submonoid."]
theorem torsion_eq_torsion_submonoid : CommMonoid.torsion G = (torsion G).toSubmonoid :=
  rfl


@[to_additive]
theorem mem_torsion (g : G) : g ∈ torsion G ↔ IsOfFinOrder g := Iff.rfl


/-- The `p`-primary component is the subgroup of elements with order prime-power of `p`. -/
@[to_additive (attr := simps!)
      "The `p`-primary component is the subgroup of elements with additive order
      prime-power of `p`."]
def primaryComponent : Subgroup G :=
  { CommMonoid.primaryComponent G p with
    inv_mem' := fun {g} ⟨n, hn⟩ => ⟨n, (orderOf_inv g).trans hn⟩ }


/-- The `p`-primary component is a `p` group. -/
theorem primaryComponent.isPGroup : IsPGroup p <| primaryComponent G p := fun g =>
  (propext exists_orderOf_eq_prime_pow_iff.symm).mpr
    (CommMonoid.primaryComponent.exists_orderOf_eq_prime_pow g)


/-- A predicate on a monoid saying that only 1 is of finite order. -/
@[to_additive "A predicate on an additive monoid saying that only 0 is of finite order."]
def IsTorsionFree :=
  ∀ g : G, g ≠ 1 → ¬IsOfFinOrder g


/-- A nontrivial monoid is not torsion-free if any nontrivial element has finite order. -/
@[to_additive (attr := simp) "An additive monoid is not torsion free if any
  nontrivial element has finite order."]
theorem not_isTorsionFree_iff : ¬IsTorsionFree G ↔ ∃ g : G, g ≠ 1 ∧ IsOfFinOrder g := by
  /-
    G : Type u_1
    inst✝ : Monoid G
    ⊢ Iff (Not (Monoid.IsTorsionFree G)) (Exists fun g => And (Ne g 1) (IsOfFinOrd …
  -/
  simp_rw [IsTorsionFree, Ne, not_forall, Classical.not_not, exists_prop]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
lemma isTorsionFree_of_subsingleton [Subsingleton G] : IsTorsionFree G :=
  fun _a ha _ => ha <| Subsingleton.elim _ _


@[to_additive]
lemma isTorsionFree_iff_torsion_eq_bot {G} [CommGroup G] :
    IsTorsionFree G ↔ CommGroup.torsion G = ⊥ := by
  /-
    G : Type u_3
    inst✝ : CommGroup G
    ⊢ Iff (Monoid.IsTorsionFree G) (Eq (CommGroup.torsion G) Bot.bot)
  -/
  rw [IsTorsionFree, eq_bot_iff, SetLike.le_def]
  /-
    G : Type u_3
    inst✝ : CommGroup G
    ⊢ Iff (∀ (g : G), Ne g 1 → Not (IsOfFinOrder g)) (∀ ⦃x : G⦄, Membership.mem (C …
  -/
  simp [not_imp_not, CommGroup.mem_torsion]
  /-
    🎉 no goals
  -/


/-- A nontrivial torsion group is not torsion-free. -/
@[to_additive "A nontrivial additive torsion group is not torsion-free."]
theorem IsTorsion.not_torsion_free [hN : Nontrivial G] : IsTorsion G → ¬IsTorsionFree G := fun tG =>
  not_isTorsionFree_iff.mpr <| by
    /-
      G : Type u_1
      inst✝ : Group G
      hN : Nontrivial G
      tG : Monoid.IsTorsion G
      ⊢ Exists fun g => And (Ne g 1) (IsOfFinOrder g)
    -/
    obtain ⟨x, hx⟩ := (nontrivial_iff_exists_ne (1 : G)).mp hN
    /-
      case intro
      G : Type u_1
      inst✝ : Group G
      hN : Nontrivial G
      tG : Monoid.IsTorsion G
      x : G
      hx : Ne x 1
      ⊢ Exists fun g => And (Ne g 1) (IsOfFinOrder g)
    -/
    exact ⟨x, hx, tG x⟩
    /-
      🎉 no goals
    -/


/-- A nontrivial torsion-free group is not torsion. -/
@[to_additive "A nontrivial torsion-free additive group is not torsion."]
theorem IsTorsionFree.not_torsion [hN : Nontrivial G] : IsTorsionFree G → ¬IsTorsion G := fun tfG =>
  (not_isTorsion_iff _).mpr <| by
    /-
      G : Type u_1
      inst✝ : Group G
      hN : Nontrivial G
      tfG : Monoid.IsTorsionFree G
      ⊢ Exists fun g => Not (IsOfFinOrder g)
    -/
    obtain ⟨x, hx⟩ := (nontrivial_iff_exists_ne (1 : G)).mp hN
    /-
      case intro
      G : Type u_1
      inst✝ : Group G
      hN : Nontrivial G
      tfG : Monoid.IsTorsionFree G
      x : G
      hx : Ne x 1
      ⊢ Exists fun g => Not (IsOfFinOrder g)
    -/
    exact ⟨x, (tfG x) hx⟩
    /-
      🎉 no goals
    -/


/-- Subgroups of torsion-free groups are torsion-free. -/
@[to_additive "Subgroups of additive torsion-free groups are additively torsion-free."]
theorem IsTorsionFree.subgroup (tG : IsTorsionFree G) (H : Subgroup G) : IsTorsionFree H :=
                                                             /-
                                                               G : Type u_1
                                                               inst✝ : Group G
                                                               tG : Monoid.IsTorsionFree G
                                                               H : Subgroup G
                                                               h : Subtype fun x => Membership.mem H x
                                                               hne : Ne h 1
                                                               ⊢ Ne (↑h) 1
                                                             -/
  fun h hne ↦ Submonoid.isOfFinOrder_coe.not.1 <| tG h <| by norm_cast
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- Direct products of torsion free groups are torsion free. -/
@[to_additive AddMonoid.IsTorsionFree.prod
      "Direct products of additive torsion free groups are torsion free."]
theorem IsTorsionFree.prod {η : Type*} {Gs : η → Type*} [∀ i, Group (Gs i)]
    (tfGs : ∀ i, IsTorsionFree (Gs i)) : IsTorsionFree <| ∀ i, Gs i := fun w hne h =>
  hne <|
    funext fun i => Classical.not_not.mp <| mt (tfGs i (w i)) <| Classical.not_not.mpr <| h.apply i


/-- Quotienting a group by its torsion subgroup yields a torsion free group. -/
@[to_additive
"Quotienting a group by its additive torsion subgroup yields an additive torsion free group."]
theorem IsTorsionFree.quotient_torsion : IsTorsionFree <| G ⧸ torsion G := fun g hne hfin =>
  hne <| by
    /-
      G : Type u_1
      inst✝ : CommGroup G
      g : HasQuotient.Quotient G (CommGroup.torsion G)
      hne : Ne g 1
      hfin : IsOfFinOrder g
      ⊢ Eq g 1
    -/
    induction' g using QuotientGroup.induction_on with g
    /-
      case H
      G : Type u_1
      inst✝ : CommGroup G
      g : G
      hne : Ne (↑g) 1
      hfin : IsOfFinOrder ↑g
      ⊢ Eq (↑g) 1
    -/
    obtain ⟨m, mpos, hm⟩ := hfin.exists_pow_eq_one
    /-
      case H.intro.intro
      G : Type u_1
      inst✝ : CommGroup G
      g : G
      hne : Ne (↑g) 1
      hfin : IsOfFinOrder ↑g
      m : Nat
      mpos : LT.lt 0 m
      hm : Eq (HPow.hPow (↑g) m) 1
      ⊢ Eq (↑g) 1
    -/
    obtain ⟨n, npos, hn⟩ := ((QuotientGroup.eq_one_iff _).mp hm).exists_pow_eq_one
    exact
      (QuotientGroup.eq_one_iff g).mpr
        (isOfFinOrder_iff_pow_eq_one.mpr ⟨m * n, mul_pos mpos npos, (pow_mul g m n).symm ▸ hn⟩)


lemma isTorsionFree_iff_noZeroSMulDivisors_nat {M : Type*} [AddMonoid M] :
    IsTorsionFree M ↔ NoZeroSMulDivisors ℕ M := by
  simp_rw [AddMonoid.IsTorsionFree, isOfFinAddOrder_iff_nsmul_eq_zero, not_exists, not_and,
    pos_iff_ne_zero, noZeroSMulDivisors_iff, forall_swap (β := ℕ)]
  /-
    M : Type u_3
    inst✝ : AddMonoid M
    ⊢ Iff (∀ (y : Nat) (x : M), Ne x 0 → Ne y 0 → Not (Eq (HSMul.hSMul y x) 0)) (∀ …
  -/
  exact forall₂_congr fun _ _ ↦ by tauto
  /-
    🎉 no goals
  -/


lemma isTorsionFree_iff_noZeroSMulDivisors_int [AddGroup G] :
    IsTorsionFree G ↔ NoZeroSMulDivisors ℤ G := by
  simp_rw [AddMonoid.IsTorsionFree, isOfFinAddOrder_iff_zsmul_eq_zero, not_exists, not_and,
    noZeroSMulDivisors_iff, forall_swap (β := ℤ)]
  /-
    G : Type u_1
    inst✝ : AddGroup G
    ⊢ Iff (∀ (y : Int) (x : G), Ne x 0 → Ne y 0 → Not (Eq (HSMul.hSMul y x) 0)) (∀ …
  -/
  exact forall₂_congr fun _ _ ↦ by tauto
  /-
    🎉 no goals
  -/


lemma IsTorsionFree.of_noZeroSMulDivisors {M : Type*} [AddMonoid M] [NoZeroSMulDivisors ℕ M] :
    IsTorsionFree M := isTorsionFree_iff_noZeroSMulDivisors_nat.2 ‹_›


alias ⟨IsTorsionFree.noZeroSMulDivisors_nat, _⟩ := isTorsionFree_iff_noZeroSMulDivisors_nat

alias ⟨IsTorsionFree.noZeroSMulDivisors_int, _⟩ := isTorsionFree_iff_noZeroSMulDivisors_int


instance {R M : Type*} [Ring R] [AddCommGroup M] [Module R M] :
    Module R (M ⧸ AddCommGroup.torsion M) :=
  letI : Submodule R M := { AddCommGroup.torsion M with smul_mem' := fun r m ⟨n, hn, hn'⟩ ↦
    ⟨n, hn, by { simp only [Function.IsPeriodicPt, Function.IsFixedPt, add_left_iterate, add_zero,
      Nat.isUnit_iff, smul_comm n] at hn' ⊢; simp only [hn', smul_zero] }⟩ }
  inferInstanceAs (Module R (M ⧸ this))



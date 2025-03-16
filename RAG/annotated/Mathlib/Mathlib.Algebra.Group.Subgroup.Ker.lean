/-- The range of a monoid homomorphism from a group is a subgroup. -/
@[to_additive "The range of an `AddMonoidHom` from an `AddGroup` is an `AddSubgroup`."]
def range (f : G →* N) : Subgroup N :=
                                                           /-
                                                             G : Type u_1
                                                             G' : Type u_2
                                                             G'' : Type u_3
                                                             inst✝⁵ : Group G
                                                             inst✝⁴ : Group G'
                                                             inst✝³ : Group G''
                                                             A : Type u_4
                                                             inst✝² : AddGroup A
                                                             N : Type u_5
                                                             P : Type u_6
                                                             inst✝¹ : Group N
                                                             inst✝ : Group P
                                                             K : Subgroup G
                                                             f : MonoidHom G N
                                                             ⊢ Eq (Set.range ⇑f) ↑(Subgroup.map f Top.top)
                                                           -/
  Subgroup.copy ((⊤ : Subgroup G).map f) (Set.range f) (by simp [Set.ext_iff])
                                                           /-
                                                             🎉 no goals
                                                           -/


@[to_additive (attr := simp)]
theorem coe_range (f : G →* N) : (f.range : Set N) = Set.range f :=
  rfl


@[to_additive (attr := simp)]
theorem mem_range {f : G →* N} {y : N} : y ∈ f.range ↔ ∃ x, f x = y :=
  Iff.rfl


@[to_additive]
                                                                           /-
                                                                             G : Type u_1
                                                                             inst✝¹ : Group G
                                                                             N : Type u_5
                                                                             inst✝ : Group N
                                                                             f : MonoidHom G N
                                                                             ⊢ Eq f.range (Subgroup.map f Top.top)
                                                                           -/
theorem range_eq_map (f : G →* N) : f.range = (⊤ : Subgroup G).map f := by ext; simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[to_additive]
instance range_isCommutative {G : Type*} [CommGroup G] {N : Type*} [Group N] (f : G →* N) :
    f.range.IsCommutative :=
  range_eq_map f ▸ Subgroup.map_isCommutative ⊤ f


@[to_additive (attr := simp)]
theorem restrict_range (f : G →* N) : (f.restrict K).range = K.map f := by
  simp_rw [SetLike.ext_iff, mem_range, mem_map, restrict_apply, SetLike.exists,
    exists_prop, forall_const]


/-- The canonical surjective group homomorphism `G →* f(G)` induced by a group
homomorphism `G →* N`. -/
@[to_additive
      "The canonical surjective `AddGroup` homomorphism `G →+ f(G)` induced by a group
      homomorphism `G →+ N`."]
def rangeRestrict (f : G →* N) : G →* f.range :=
  codRestrict f _ fun x => ⟨x, rfl⟩


@[to_additive (attr := simp)]
theorem coe_rangeRestrict (f : G →* N) (g : G) : (f.rangeRestrict g : N) = f g :=
  rfl


@[to_additive]
theorem coe_comp_rangeRestrict (f : G →* N) :
    ((↑) : f.range → N) ∘ (⇑f.rangeRestrict : G → f.range) = f :=
  rfl


@[to_additive]
theorem subtype_comp_rangeRestrict (f : G →* N) : f.range.subtype.comp f.rangeRestrict = f :=
  ext <| f.coe_rangeRestrict


@[to_additive]
theorem rangeRestrict_surjective (f : G →* N) : Function.Surjective f.rangeRestrict :=
  fun ⟨_, g, rfl⟩ => ⟨g, rfl⟩


@[to_additive (attr := simp)]
lemma rangeRestrict_injective_iff {f : G →* N} : Injective f.rangeRestrict ↔ Injective f := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    ⊢ Iff (Function.Injective ⇑f.rangeRestrict) (Function.Injective ⇑f)
  -/
  convert Set.injective_codRestrict _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem map_range (g : N →* P) (f : G →* N) : f.range.map g = (g.comp f).range := by
  /-
    G : Type u_1
    inst✝² : Group G
    N : Type u_5
    P : Type u_6
    inst✝¹ : Group N
    inst✝ : Group P
    g : MonoidHom N P
    f : MonoidHom G N
    ⊢ Eq (Subgroup.map g f.range) (g.comp f).range
  -/
  rw [range_eq_map, range_eq_map]; exact (⊤ : Subgroup G).map_map g f
                                   /-
                                     🎉 no goals
                                   -/


@[to_additive]
lemma range_comp (g : N →* P) (f : G →* N) : (g.comp f).range = f.range.map g := (map_range ..).symm


@[to_additive]
theorem range_eq_top {N} [Group N] {f : G →* N} :
    f.range = (⊤ : Subgroup N) ↔ Function.Surjective f :=
                                          /-
                                            G : Type u_1
                                            inst✝¹ : Group G
                                            N : Type u_7
                                            inst✝ : Group N
                                            f : MonoidHom G N
                                            ⊢ Iff (Eq ↑f.range ↑Top.top) (Eq (Set.range ⇑f) Set.univ)
                                          -/
  SetLike.ext'_iff.trans <| Iff.trans (by rw [coe_range, coe_top]) Set.range_eq_univ
                                          /-
                                            🎉 no goals
                                          -/


@[deprecated (since := "2024-11-11")] alias range_top_iff_surjective := range_eq_top


/-- The range of a surjective monoid homomorphism is the whole of the codomain. -/
@[to_additive (attr := simp)
  "The range of a surjective `AddMonoid` homomorphism is the whole of the codomain."]
theorem range_eq_top_of_surjective {N} [Group N] (f : G →* N) (hf : Function.Surjective f) :
    f.range = (⊤ : Subgroup N) :=
  range_eq_top.2 hf


@[deprecated (since := "2024-11-11")] alias range_top_of_surjective := range_eq_top_of_surjective


@[to_additive (attr := simp)]
theorem range_one : (1 : G →* N).range = ⊥ :=
                          /-
                            G : Type u_1
                            inst✝¹ : Group G
                            N : Type u_5
                            inst✝ : Group N
                            x : N
                            ⊢ Iff (Membership.mem (MonoidHom.range 1) x) (Membership.mem Bot.bot x)
                          -/
  SetLike.ext fun x => by simpa using @comm _ (· = ·) _ 1 x
                          /-
                            🎉 no goals
                          -/


@[to_additive (attr := simp)]
theorem _root_.Subgroup.range_subtype (H : Subgroup G) : H.subtype.range = H :=
  SetLike.coe_injective <| (coe_range _).trans <| Subtype.range_coe


@[to_additive]
alias _root_.Subgroup.subtype_range := Subgroup.range_subtype

-- `alias` doesn't add the deprecation suggestion to the `to_additive` version
-- see https://github.com/leanprover-community/mathlib4/issues/19424

@[to_additive (attr := simp)]
theorem _root_.Subgroup.inclusion_range {H K : Subgroup G} (h_le : H ≤ K) :
    (inclusion h_le).range = H.subgroupOf K :=
  Subgroup.ext fun g => Set.ext_iff.mp (Set.range_inclusion h_le) g


@[to_additive]
theorem subgroupOf_range_eq_of_le {G₁ G₂ : Type*} [Group G₁] [Group G₂] {K : Subgroup G₂}
    (f : G₁ →* G₂) (h : f.range ≤ K) :
    f.range.subgroupOf K = (f.codRestrict K fun x => h ⟨x, rfl⟩).range := by
  /-
    G₁ : Type u_7
    G₂ : Type u_8
    inst✝¹ : Group G₁
    inst✝ : Group G₂
    K : Subgroup G₂
    f : MonoidHom G₁ G₂
    h : LE.le f.range K
    ⊢ Eq (f.range.subgroupOf K) (f.codRestrict K ⋯).range
  -/
  ext k
  /-
    case h
    G₁ : Type u_7
    G₂ : Type u_8
    inst✝¹ : Group G₁
    inst✝ : Group G₂
    K : Subgroup G₂
    f : MonoidHom G₁ G₂
    h : LE.le f.range K
    k : Subtype fun x => Membership.mem K x
    ⊢ Iff (Membership.mem (f.range.subgroupOf K) k) (Membership.mem (f.codRestrict …
  -/
  refine exists_congr ?_
  /-
    case h
    G₁ : Type u_7
    G₂ : Type u_8
    inst✝¹ : Group G₁
    inst✝ : Group G₂
    K : Subgroup G₂
    f : MonoidHom G₁ G₂
    h : LE.le f.range K
    k : Subtype fun x => Membership.mem K x
    ⊢ ∀ (a : G₁), Iff (Eq (f a) (K.subtype k)) (Eq ((f.codRestrict K ⋯) a) k)
  -/
  simp [Subtype.ext_iff]
  /-
    🎉 no goals
  -/


/-- Computable alternative to `MonoidHom.ofInjective`. -/
@[to_additive "Computable alternative to `AddMonoidHom.ofInjective`."]
def ofLeftInverse {f : G →* N} {g : N →* G} (h : Function.LeftInverse g f) : G ≃* f.range :=
  { f.rangeRestrict with
    toFun := f.rangeRestrict
    invFun := g ∘ f.range.subtype
    left_inv := h
    right_inv := by
      /-
        G : Type u_1
        G' : Type u_2
        G'' : Type u_3
        inst✝⁵ : Group G
        inst✝⁴ : Group G'
        inst✝³ : Group G''
        A : Type u_4
        inst✝² : AddGroup A
        N : Type u_5
        P : Type u_6
        inst✝¹ : Group N
        inst✝ : Group P
        K : Subgroup G
        f : MonoidHom G N
        g : MonoidHom N G
        h : Function.LeftInverse ⇑g ⇑f
        ⊢ Function.RightInverse (Function.comp ⇑g ⇑f.range.subtype) ⇑f.rangeRestrict
      -/
      rintro ⟨x, y, rfl⟩
      /-
        case mk.intro
        G : Type u_1
        G' : Type u_2
        G'' : Type u_3
        inst✝⁵ : Group G
        inst✝⁴ : Group G'
        inst✝³ : Group G''
        A : Type u_4
        inst✝² : AddGroup A
        N : Type u_5
        P : Type u_6
        inst✝¹ : Group N
        inst✝ : Group P
        K : Subgroup G
        f : MonoidHom G N
        g : MonoidHom N G
        h : Function.LeftInverse ⇑g ⇑f
        y : G
        ⊢ Eq (f.rangeRestrict (Function.comp ⇑g ⇑f.range.subtype ⟨f y, ⋯⟩)) ⟨f y, ⋯⟩
      -/
      apply Subtype.ext
      /-
        case mk.intro.a
        G : Type u_1
        G' : Type u_2
        G'' : Type u_3
        inst✝⁵ : Group G
        inst✝⁴ : Group G'
        inst✝³ : Group G''
        A : Type u_4
        inst✝² : AddGroup A
        N : Type u_5
        P : Type u_6
        inst✝¹ : Group N
        inst✝ : Group P
        K : Subgroup G
        f : MonoidHom G N
        g : MonoidHom N G
        h : Function.LeftInverse ⇑g ⇑f
        y : G
        ⊢ Eq ↑(f.rangeRestrict (Function.comp ⇑g ⇑f.range.subtype ⟨f y, ⋯⟩)) ↑⟨f y, ⋯⟩
      -/
      rw [coe_rangeRestrict, Function.comp_apply, Subgroup.coeSubtype, Subtype.coe_mk, h] }
      /-
        🎉 no goals
      -/


@[to_additive (attr := simp)]
theorem ofLeftInverse_apply {f : G →* N} {g : N →* G} (h : Function.LeftInverse g f) (x : G) :
    ↑(ofLeftInverse h x) = f x :=
  rfl


@[to_additive (attr := simp)]
theorem ofLeftInverse_symm_apply {f : G →* N} {g : N →* G} (h : Function.LeftInverse g f)
    (x : f.range) : (ofLeftInverse h).symm x = g x :=
  rfl


/-- The range of an injective group homomorphism is isomorphic to its domain. -/
@[to_additive "The range of an injective additive group homomorphism is isomorphic to its
domain."]
noncomputable def ofInjective {f : G →* N} (hf : Function.Injective f) : G ≃* f.range :=
  MulEquiv.ofBijective (f.codRestrict f.range fun x => ⟨x, rfl⟩)
    ⟨fun _ _ h => hf (Subtype.ext_iff.mp h), by
      /-
        G : Type u_1
        G' : Type u_2
        G'' : Type u_3
        inst✝⁵ : Group G
        inst✝⁴ : Group G'
        inst✝³ : Group G''
        A : Type u_4
        inst✝² : AddGroup A
        N : Type u_5
        P : Type u_6
        inst✝¹ : Group N
        inst✝ : Group P
        K : Subgroup G
        f : MonoidHom G N
        hf : Function.Injective ⇑f
        ⊢ Function.Surjective ⇑(f.codRestrict f.range ⋯)
      -/
      rintro ⟨x, y, rfl⟩
      /-
        case mk.intro
        G : Type u_1
        G' : Type u_2
        G'' : Type u_3
        inst✝⁵ : Group G
        inst✝⁴ : Group G'
        inst✝³ : Group G''
        A : Type u_4
        inst✝² : AddGroup A
        N : Type u_5
        P : Type u_6
        inst✝¹ : Group N
        inst✝ : Group P
        K : Subgroup G
        f : MonoidHom G N
        hf : Function.Injective ⇑f
        y : G
        ⊢ Exists fun a => Eq ((f.codRestrict f.range ⋯) a) ⟨f y, ⋯⟩
      -/
      exact ⟨y, rfl⟩⟩
      /-
        🎉 no goals
      -/


@[to_additive]
theorem ofInjective_apply {f : G →* N} (hf : Function.Injective f) {x : G} :
    ↑(ofInjective hf x) = f x :=
  rfl


@[to_additive (attr := simp)]
theorem apply_ofInjective_symm {f : G →* N} (hf : Function.Injective f) (x : f.range) :
    f ((ofInjective hf).symm x) = x :=
  Subtype.ext_iff.1 <| (ofInjective hf).apply_symm_apply x


@[simp]
theorem coe_toAdditive_range (f : G →* G') :
    (MonoidHom.toAdditive f).range = Subgroup.toAddSubgroup f.range := rfl


@[simp]
theorem coe_toMultiplicative_range {A A' : Type*} [AddGroup A] [AddGroup A'] (f : A →+ A') :
    (AddMonoidHom.toMultiplicative f).range = AddSubgroup.toSubgroup f.range := rfl


/-- The multiplicative kernel of a monoid homomorphism is the subgroup of elements `x : G` such that
`f x = 1` -/
@[to_additive
      "The additive kernel of an `AddMonoid` homomorphism is the `AddSubgroup` of elements
      such that `f x = 0`"]
def ker (f : G →* M) : Subgroup G :=
  { MonoidHom.mker f with
    inv_mem' := fun {x} (hx : f x = 1) =>
      calc
                                  /-
                                    G : Type u_1
                                    G' : Type u_2
                                    G'' : Type u_3
                                    inst✝⁶ : Group G
                                    inst✝⁵ : Group G'
                                    inst✝⁴ : Group G''
                                    A : Type u_4
                                    inst✝³ : AddGroup A
                                    N : Type u_5
                                    P : Type u_6
                                    inst✝² : Group N
                                    inst✝¹ : Group P
                                    K : Subgroup G
                                    M : Type u_7
                                    inst✝ : MulOneClass M
                                    f : MonoidHom G M
                                    x : G
                                    hx : Eq (f x) 1
                                    ⊢ Eq (f (Inv.inv x)) (HMul.hMul (f x) (f (Inv.inv x)))
                                  -/
        f x⁻¹ = f x * f x⁻¹ := by rw [hx, one_mul]
                                  /-
                                    🎉 no goals
                                  -/
                    /-
                      G : Type u_1
                      G' : Type u_2
                      G'' : Type u_3
                      inst✝⁶ : Group G
                      inst✝⁵ : Group G'
                      inst✝⁴ : Group G''
                      A : Type u_4
                      inst✝³ : AddGroup A
                      N : Type u_5
                      P : Type u_6
                      inst✝² : Group N
                      inst✝¹ : Group P
                      K : Subgroup G
                      M : Type u_7
                      inst✝ : MulOneClass M
                      f : MonoidHom G M
                      x : G
                      hx : Eq (f x) 1
                      ⊢ Eq (HMul.hMul (f x) (f (Inv.inv x))) 1
                    -/
        _ = 1 := by rw [← map_mul, mul_inv_cancel, map_one] }
                    /-
                      🎉 no goals
                    -/


@[to_additive (attr := simp)]
theorem mem_ker {f : G →* M} {x : G} : x ∈ f.ker ↔ f x = 1 :=
  Iff.rfl


@[to_additive]
theorem div_mem_ker_iff (f : G →* N) {x y : G} : x / y ∈ ker f ↔ f x = f y := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    x y : G
    ⊢ Iff (Membership.mem f.ker (HDiv.hDiv x y)) (Eq (f x) (f y))
  -/
  rw [mem_ker, map_div, div_eq_one]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem coe_ker (f : G →* M) : (f.ker : Set G) = (f : G → M) ⁻¹' {1} :=
  rfl


@[to_additive (attr := simp)]
theorem ker_toHomUnits {M} [Monoid M] (f : G →* M) : f.toHomUnits.ker = f.ker := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    M : Type u_8
    inst✝ : Monoid M
    f : MonoidHom G M
    ⊢ Eq f.toHomUnits.ker f.ker
  -/
  ext x
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    M : Type u_8
    inst✝ : Monoid M
    f : MonoidHom G M
    x : G
    ⊢ Iff (Membership.mem f.toHomUnits.ker x) (Membership.mem f.ker x)
  -/
  simp [mem_ker, Units.ext_iff]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem eq_iff (f : G →* M) {x y : G} : f x = f y ↔ y⁻¹ * x ∈ f.ker := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    M : Type u_7
    inst✝ : MulOneClass M
    f : MonoidHom G M
    x y : G
    ⊢ Iff (Eq (f x) (f y)) (Membership.mem f.ker (HMul.hMul (Inv.inv y) x))
  -/
  constructor <;> intro h
    /-
      case mp
      G : Type u_1
      inst✝¹ : Group G
      M : Type u_7
      inst✝ : MulOneClass M
      f : MonoidHom G M
      x y : G
      h : Eq (f x) (f y)
      ⊢ Membership.mem f.ker (HMul.hMul (Inv.inv y) x)
    -/
  · rw [mem_ker, map_mul, h, ← map_mul, inv_mul_cancel, map_one]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      G : Type u_1
      inst✝¹ : Group G
      M : Type u_7
      inst✝ : MulOneClass M
      f : MonoidHom G M
      x y : G
      h : Membership.mem f.ker (HMul.hMul (Inv.inv y) x)
      ⊢ Eq (f x) (f y)
    -/
  · rw [← one_mul x, ← mul_inv_cancel y, mul_assoc, map_mul, mem_ker.1 h, mul_one]
    /-
      🎉 no goals
    -/


@[to_additive]
instance decidableMemKer [DecidableEq M] (f : G →* M) : DecidablePred (· ∈ f.ker) := fun x =>
  decidable_of_iff (f x = 1) f.mem_ker


@[to_additive]
theorem comap_ker (g : N →* P) (f : G →* N) : g.ker.comap f = (g.comp f).ker :=
  rfl


@[to_additive (attr := simp)]
theorem comap_bot (f : G →* N) : (⊥ : Subgroup N).comap f = f.ker :=
  rfl


@[to_additive (attr := simp)]
theorem ker_restrict (f : G →* N) : (f.restrict K).ker = f.ker.subgroupOf K :=
  rfl


@[to_additive (attr := simp)]
theorem ker_codRestrict {S} [SetLike S N] [SubmonoidClass S N] (f : G →* N) (s : S)
    (h : ∀ x, f x ∈ s) : (f.codRestrict s h).ker = f.ker :=
  SetLike.ext fun _x => Subtype.ext_iff


@[to_additive (attr := simp)]
theorem ker_rangeRestrict (f : G →* N) : ker (rangeRestrict f) = ker f :=
  ker_codRestrict _ _ _


@[to_additive (attr := simp)]
theorem ker_one : (1 : G →* M).ker = ⊤ :=
  SetLike.ext fun _x => eq_self_iff_true _


@[to_additive (attr := simp)]
theorem ker_id : (MonoidHom.id G).ker = ⊥ :=
  rfl


@[to_additive]
theorem ker_eq_bot_iff (f : G →* M) : f.ker = ⊥ ↔ Function.Injective f :=
                       /-
                         G : Type u_1
                         inst✝¹ : Group G
                         M : Type u_7
                         inst✝ : MulOneClass M
                         f : MonoidHom G M
                         h : Eq f.ker Bot.bot
                         x y : G
                         hxy : Eq (f x) (f y)
                         ⊢ Eq x y
                       -/
  ⟨fun h x y hxy => by rwa [eq_iff, h, mem_bot, inv_mul_eq_one, eq_comm] at hxy, fun h =>
                       /-
                         🎉 no goals
                       -/
    bot_unique fun _ hx => h (hx.trans f.map_one.symm)⟩


@[to_additive (attr := simp)]
theorem _root_.Subgroup.ker_subtype (H : Subgroup G) : H.subtype.ker = ⊥ :=
  H.subtype.ker_eq_bot_iff.mpr Subtype.coe_injective


@[to_additive (attr := simp)]
theorem _root_.Subgroup.ker_inclusion {H K : Subgroup G} (h : H ≤ K) : (inclusion h).ker = ⊥ :=
  (inclusion h).ker_eq_bot_iff.mpr (Set.inclusion_injective h)


@[to_additive]
theorem ker_prod {M N : Type*} [MulOneClass M] [MulOneClass N] (f : G →* M) (g : G →* N) :
    (f.prod g).ker = f.ker ⊓ g.ker :=
  SetLike.ext fun _ => Prod.mk_eq_one


@[to_additive]
theorem range_le_ker_iff (f : G →* G') (g : G' →* G'') : f.range ≤ g.ker ↔ g.comp f = 1 :=
                                        /-
                                          G : Type u_1
                                          G' : Type u_2
                                          G'' : Type u_3
                                          inst✝² : Group G
                                          inst✝¹ : Group G'
                                          inst✝ : Group G''
                                          f : MonoidHom G G'
                                          g : MonoidHom G' G''
                                          ⊢ Eq (g.comp f) 1 → LE.le f.range g.ker
                                        -/
  ⟨fun h => ext fun x => h ⟨x, rfl⟩, by rintro h _ ⟨y, rfl⟩; exact DFunLike.congr_fun h y⟩
                                                             /-
                                                               🎉 no goals
                                                             -/


@[to_additive]
instance (priority := 100) normal_ker (f : G →* M) : f.ker.Normal :=
  ⟨fun x hx y => by
    /-
      G : Type u_1
      G' : Type u_2
      G'' : Type u_3
      inst✝⁶ : Group G
      inst✝⁵ : Group G'
      inst✝⁴ : Group G''
      A : Type u_4
      inst✝³ : AddGroup A
      N : Type u_5
      P : Type u_6
      inst✝² : Group N
      inst✝¹ : Group P
      K : Subgroup G
      M : Type u_7
      inst✝ : MulOneClass M
      f : MonoidHom G M
      x : G
      hx : Membership.mem f.ker x
      y : G
      ⊢ Membership.mem f.ker (HMul.hMul (HMul.hMul y x) (Inv.inv y))
    -/
    rw [mem_ker, map_mul, map_mul, mem_ker.1 hx, mul_one, map_mul_eq_one f (mul_inv_cancel y)]⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_toAdditive_ker (f : G →* G') :
    (MonoidHom.toAdditive f).ker = Subgroup.toAddSubgroup f.ker := rfl


@[simp]
theorem coe_toMultiplicative_ker {A A' : Type*} [AddGroup A] [AddGroup A'] (f : A →+ A') :
    (AddMonoidHom.toMultiplicative f).ker = AddSubgroup.toSubgroup f.ker := rfl


/-- The subgroup of elements `x : G` such that `f x = g x` -/
@[to_additive "The additive subgroup of elements `x : G` such that `f x = g x`"]
def eqLocus (f g : G →* M) : Subgroup G :=
  { eqLocusM f g with inv_mem' := eq_on_inv f g }


@[to_additive (attr := simp)]
theorem eqLocus_same (f : G →* N) : f.eqLocus f = ⊤ :=
  SetLike.ext fun _ => eq_self_iff_true _


/-- If two monoid homomorphisms are equal on a set, then they are equal on its subgroup closure. -/
@[to_additive
      "If two monoid homomorphisms are equal on a set, then they are equal on its subgroup
      closure."]
theorem eqOn_closure {f g : G →* M} {s : Set G} (h : Set.EqOn f g s) : Set.EqOn f g (closure s) :=
  show closure s ≤ f.eqLocus g from (closure_le _).2 h


@[to_additive]
theorem eq_of_eqOn_top {f g : G →* M} (h : Set.EqOn f g (⊤ : Subgroup G)) : f = g :=
  ext fun _x => h trivial


@[to_additive]
theorem eq_of_eqOn_dense {s : Set G} (hs : closure s = ⊤) {f g : G →* M} (h : s.EqOn f g) : f = g :=
  eq_of_eqOn_top <| hs ▸ eqOn_closure h


@[to_additive]
theorem map_eq_bot_iff {f : G →* N} : H.map f = ⊥ ↔ H ≤ f.ker :=
  (gc_map_comap f).l_eq_bot


@[to_additive]
theorem map_eq_bot_iff_of_injective {f : G →* N} (hf : Function.Injective f) :
                              /-
                                G : Type u_1
                                inst✝¹ : Group G
                                N : Type u_5
                                inst✝ : Group N
                                H : Subgroup G
                                f : MonoidHom G N
                                hf : Function.Injective ⇑f
                                ⊢ Iff (Eq (Subgroup.map f H) Bot.bot) (Eq H Bot.bot)
                              -/
    H.map f = ⊥ ↔ H = ⊥ := by rw [map_eq_bot_iff, f.ker_eq_bot_iff.mpr hf, le_bot_iff]
                              /-
                                🎉 no goals
                              -/


@[to_additive]
theorem map_le_range (H : Subgroup G) : map f H ≤ f.range :=
  (range_eq_map f).symm ▸ map_mono le_top


@[to_additive]
theorem map_subtype_le {H : Subgroup G} (K : Subgroup H) : K.map H.subtype ≤ H :=
  (K.map_le_range H.subtype).trans_eq H.range_subtype


@[to_additive]
theorem ker_le_comap (H : Subgroup N) : f.ker ≤ comap f H :=
  comap_bot f ▸ comap_mono bot_le


@[to_additive]
theorem map_comap_eq (H : Subgroup N) : map f (comap f H) = f.range ⊓ H :=
  SetLike.ext' <| by
    /-
      G : Type u_1
      inst✝¹ : Group G
      N : Type u_5
      inst✝ : Group N
      f : MonoidHom G N
      H : Subgroup N
      ⊢ Eq ↑(Subgroup.map f (Subgroup.comap f H)) ↑(Min.min f.range H)
    -/
    rw [coe_map, coe_comap, Set.image_preimage_eq_inter_range, coe_inf, coe_range, Set.inter_comm]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem comap_map_eq (H : Subgroup G) : comap f (map f H) = H ⊔ f.ker := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    H : Subgroup G
    ⊢ Eq (Subgroup.comap f (Subgroup.map f H)) (Max.max H f.ker)
  -/
  refine le_antisymm ?_ (sup_le (le_comap_map _ _) (ker_le_comap _ _))
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    H : Subgroup G
    ⊢ LE.le (Subgroup.comap f (Subgroup.map f H)) (Max.max H f.ker)
  -/
  intro x hx; simp only [exists_prop, mem_map, mem_comap] at hx
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    H : Subgroup G
    x : G
    hx : Exists fun x_1 => And (Membership.mem H x_1) (Eq (f x_1) (f x))
    ⊢ Membership.mem (Max.max H f.ker) x
  -/
  rcases hx with ⟨y, hy, hy'⟩
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    H : Subgroup G
    x y : G
    hy : Membership.mem H y
    hy' : Eq (f y) (f x)
    ⊢ Membership.mem (Max.max H f.ker) x
  -/
  rw [← mul_inv_cancel_left y x]
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    H : Subgroup G
    x y : G
    hy : Membership.mem H y
    hy' : Eq (f y) (f x)
    ⊢ Membership.mem (Max.max H f.ker) (HMul.hMul y (HMul.hMul (Inv.inv y) x))
  -/
  exact mul_mem_sup hy (by simp [mem_ker, hy'])
  /-
    🎉 no goals
  -/


@[to_additive]
theorem map_comap_eq_self {f : G →* N} {H : Subgroup N} (h : H ≤ f.range) :
    map f (comap f H) = H := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    H : Subgroup N
    h : LE.le H f.range
    ⊢ Eq (Subgroup.map f (Subgroup.comap f H)) H
  -/
  rwa [map_comap_eq, inf_eq_right]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem map_comap_eq_self_of_surjective {f : G →* N} (h : Function.Surjective f) (H : Subgroup N) :
    map f (comap f H) = H :=
  map_comap_eq_self (range_eq_top.2 h ▸ le_top)


@[to_additive]
theorem comap_le_comap_of_le_range {f : G →* N} {K L : Subgroup N} (hf : K ≤ f.range) :
    K.comap f ≤ L.comap f ↔ K ≤ L :=
  ⟨(map_comap_eq_self hf).ge.trans ∘ map_le_iff_le_comap.mpr, comap_mono⟩


@[to_additive]
theorem comap_le_comap_of_surjective {f : G →* N} {K L : Subgroup N} (hf : Function.Surjective f) :
    K.comap f ≤ L.comap f ↔ K ≤ L :=
  comap_le_comap_of_le_range (range_eq_top.2 hf ▸ le_top)


@[to_additive]
theorem comap_lt_comap_of_surjective {f : G →* N} {K L : Subgroup N} (hf : Function.Surjective f) :
                                        /-
                                          G : Type u_1
                                          inst✝¹ : Group G
                                          N : Type u_5
                                          inst✝ : Group N
                                          f : MonoidHom G N
                                          K L : Subgroup N
                                          hf : Function.Surjective ⇑f
                                          ⊢ Iff (LT.lt (Subgroup.comap f K) (Subgroup.comap f L)) (LT.lt K L)
                                        -/
    K.comap f < L.comap f ↔ K < L := by simp_rw [lt_iff_le_not_le, comap_le_comap_of_surjective hf]
                                        /-
                                          🎉 no goals
                                        -/


@[to_additive]
theorem comap_injective {f : G →* N} (h : Function.Surjective f) : Function.Injective (comap f) :=
                /-
                  G : Type u_1
                  inst✝¹ : Group G
                  N : Type u_5
                  inst✝ : Group N
                  f : MonoidHom G N
                  h : Function.Surjective ⇑f
                  K L : Subgroup N
                  ⊢ Eq (Subgroup.comap f K) (Subgroup.comap f L) → Eq K L
                -/
  fun K L => by simp only [le_antisymm_iff, comap_le_comap_of_surjective h, imp_self]
                /-
                  🎉 no goals
                -/


@[to_additive]
theorem comap_map_eq_self {f : G →* N} {H : Subgroup G} (h : f.ker ≤ H) :
    comap f (map f H) = H := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    H : Subgroup G
    h : LE.le f.ker H
    ⊢ Eq (Subgroup.comap f (Subgroup.map f H)) H
  -/
  rwa [comap_map_eq, sup_eq_left]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem comap_map_eq_self_of_injective {f : G →* N} (h : Function.Injective f) (H : Subgroup G) :
    comap f (map f H) = H :=
  comap_map_eq_self (((ker_eq_bot_iff _).mpr h).symm ▸ bot_le)


@[to_additive]
theorem map_le_map_iff {f : G →* N} {H K : Subgroup G} : H.map f ≤ K.map f ↔ H ≤ K ⊔ f.ker := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    H K : Subgroup G
    ⊢ Iff (LE.le (Subgroup.map f H) (Subgroup.map f K)) (LE.le H (Max.max K f.ker))
  -/
  rw [map_le_iff_le_comap, comap_map_eq]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem map_le_map_iff' {f : G →* N} {H K : Subgroup G} :
    H.map f ≤ K.map f ↔ H ⊔ f.ker ≤ K ⊔ f.ker := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    H K : Subgroup G
    ⊢ Iff (LE.le (Subgroup.map f H) (Subgroup.map f K)) (LE.le (Max.max H f.ker) ( …
  -/
  simp only [map_le_map_iff, sup_le_iff, le_sup_right, and_true]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem map_eq_map_iff {f : G →* N} {H K : Subgroup G} :
                                                    /-
                                                      G : Type u_1
                                                      inst✝¹ : Group G
                                                      N : Type u_5
                                                      inst✝ : Group N
                                                      f : MonoidHom G N
                                                      H K : Subgroup G
                                                      ⊢ Iff (Eq (Subgroup.map f H) (Subgroup.map f K)) (Eq (Max.max H f.ker) (Max.ma …
                                                    -/
    H.map f = K.map f ↔ H ⊔ f.ker = K ⊔ f.ker := by simp only [le_antisymm_iff, map_le_map_iff']
                                                    /-
                                                      🎉 no goals
                                                    -/


@[to_additive]
theorem map_eq_range_iff {f : G →* N} {H : Subgroup G} :
    H.map f = f.range ↔ Codisjoint H f.ker := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    H : Subgroup G
    ⊢ Iff (Eq (Subgroup.map f H) f.range) (Codisjoint H f.ker)
  -/
  rw [f.range_eq_map, map_eq_map_iff, codisjoint_iff, top_sup_eq]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem map_le_map_iff_of_injective {f : G →* N} (hf : Function.Injective f) {H K : Subgroup G} :
                                    /-
                                      G : Type u_1
                                      inst✝¹ : Group G
                                      N : Type u_5
                                      inst✝ : Group N
                                      f : MonoidHom G N
                                      hf : Function.Injective ⇑f
                                      H K : Subgroup G
                                      ⊢ Iff (LE.le (Subgroup.map f H) (Subgroup.map f K)) (LE.le H K)
                                    -/
    H.map f ≤ K.map f ↔ H ≤ K := by rw [map_le_iff_le_comap, comap_map_eq_self_of_injective hf]
                                    /-
                                      🎉 no goals
                                    -/


@[to_additive (attr := simp)]
theorem map_subtype_le_map_subtype {G' : Subgroup G} {H K : Subgroup G'} :
    H.map G'.subtype ≤ K.map G'.subtype ↔ H ≤ K :=
  map_le_map_iff_of_injective G'.subtype_injective


@[to_additive]
theorem map_lt_map_iff_of_injective {f : G →* N} (hf : Function.Injective f) {H K : Subgroup G} :
    H.map f < K.map f ↔ H < K :=
  lt_iff_lt_of_le_iff_le' (map_le_map_iff_of_injective hf) (map_le_map_iff_of_injective hf)


@[to_additive (attr := simp)]
theorem map_subtype_lt_map_subtype {G' : Subgroup G} {H K : Subgroup G'} :
    H.map G'.subtype < K.map G'.subtype ↔ H < K :=
  map_lt_map_iff_of_injective G'.subtype_injective


@[to_additive]
theorem map_injective {f : G →* N} (h : Function.Injective f) : Function.Injective (map f) :=
  Function.LeftInverse.injective <| comap_map_eq_self_of_injective h


/-- Given `f(A) = f(B)`, `ker f ≤ A`, and `ker f ≤ B`, deduce that `A = B`. -/
@[to_additive "Given `f(A) = f(B)`, `ker f ≤ A`, and `ker f ≤ B`, deduce that `A = B`."]
theorem map_injective_of_ker_le {H K : Subgroup G} (hH : f.ker ≤ H) (hK : f.ker ≤ K)
    (hf : map f H = map f K) : H = K := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    H K : Subgroup G
    hH : LE.le f.ker H
    hK : LE.le f.ker K
    hf : Eq (Subgroup.map f H) (Subgroup.map f K)
    ⊢ Eq H K
  -/
  apply_fun comap f at hf
  /-
    G : Type u_1
    inst✝¹ : Group G
    N : Type u_5
    inst✝ : Group N
    f : MonoidHom G N
    H K : Subgroup G
    hH : LE.le f.ker H
    hK : LE.le f.ker K
    hf : Eq (Subgroup.comap f (Subgroup.map f H)) (Subgroup.comap f (Subgroup.map  …
    ⊢ Eq H K
  -/
  rwa [comap_map_eq, comap_map_eq, sup_of_le_left hH, sup_of_le_left hK] at hf
  /-
    🎉 no goals
  -/


@[to_additive]
theorem ker_subgroupMap : (f.subgroupMap H).ker = f.ker.subgroupOf H :=
  ext fun _ ↦ Subtype.ext_iff


@[to_additive]
theorem closure_preimage_eq_top (s : Set G) : closure ((closure s).subtype ⁻¹' s) = ⊤ := by
  /-
    G : Type u_1
    inst✝ : Group G
    s : Set G
    ⊢ Eq (Subgroup.closure (Set.preimage (⇑(Subgroup.closure s).subtype) s)) Top.top
  -/
  apply map_injective (closure s).subtype_injective
  rw [MonoidHom.map_closure, ← MonoidHom.range_eq_map, range_subtype,
    Set.image_preimage_eq_of_subset]
  /-
    case a
    G : Type u_1
    inst✝ : Group G
    s : Set G
    ⊢ HasSubset.Subset s (Set.range ⇑(Subgroup.closure s).subtype)
  -/
  rw [coeSubtype, Subtype.range_coe_subtype]
  /-
    case a
    G : Type u_1
    inst✝ : Group G
    s : Set G
    ⊢ HasSubset.Subset s (setOf fun x => Membership.mem (Subgroup.closure s) x)
  -/
  exact subset_closure
  /-
    🎉 no goals
  -/


@[to_additive]
theorem comap_sup_eq_of_le_range {H K : Subgroup N} (hH : H ≤ f.range) (hK : K ≤ f.range) :
    comap f H ⊔ comap f K = comap f (H ⊔ K) :=
  map_injective_of_ker_le f ((ker_le_comap f H).trans le_sup_left) (ker_le_comap f (H ⊔ K))
    (by
      rw [map_comap_eq, map_sup, map_comap_eq, map_comap_eq, inf_eq_right.mpr hH,
        inf_eq_right.mpr hK, inf_eq_right.mpr (sup_le hH hK)])


@[to_additive]
theorem comap_sup_eq (H K : Subgroup N) (hf : Function.Surjective f) :
    comap f H ⊔ comap f K = comap f (H ⊔ K) :=
  comap_sup_eq_of_le_range f (range_eq_top.2 hf ▸ le_top) (range_eq_top.2 hf ▸ le_top)


@[to_additive]
theorem sup_subgroupOf_eq {H K L : Subgroup G} (hH : H ≤ L) (hK : K ≤ L) :
    H.subgroupOf L ⊔ K.subgroupOf L = (H ⊔ K).subgroupOf L :=
  comap_sup_eq_of_le_range L.subtype (hH.trans_eq L.range_subtype.symm)
     (hK.trans_eq L.range_subtype.symm)


@[to_additive]
theorem codisjoint_subgroupOf_sup (H K : Subgroup G) :
    Codisjoint (H.subgroupOf (H ⊔ K)) (K.subgroupOf (H ⊔ K)) := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    ⊢ Codisjoint (H.subgroupOf (Max.max H K)) (K.subgroupOf (Max.max H K))
  -/
  rw [codisjoint_iff, sup_subgroupOf_eq, subgroupOf_self]
  /-
    case hH
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    ⊢ LE.le H (Max.max H K)
  -/
  exacts [le_sup_left, le_sup_right]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem subgroupOf_sup (A A' B : Subgroup G) (hA : A ≤ B) (hA' : A' ≤ B) :
    (A ⊔ A').subgroupOf B = A.subgroupOf B ⊔ A'.subgroupOf B := by
  refine
    map_injective_of_ker_le B.subtype (ker_le_comap _ _)
      (le_trans (ker_le_comap B.subtype _) le_sup_left) ?_
  /-
    G : Type u_1
    inst✝ : Group G
    A A' B : Subgroup G
    hA : LE.le A B
    hA' : LE.le A' B
    ⊢ Eq (Subgroup.map B.subtype ((Max.max A A').subgroupOf B)) (Subgroup.map B.su …
  -/
  simp only [subgroupOf, map_comap_eq, map_sup, range_subtype]
  /-
    G : Type u_1
    inst✝ : Group G
    A A' B : Subgroup G
    hA : LE.le A B
    hA' : LE.le A' B
    ⊢ Eq (Min.min B (Max.max A A')) (Max.max (Min.min B A) (Min.min B A'))
  -/
  rw [inf_of_le_right (sup_le hA hA'), inf_of_le_right hA', inf_of_le_right hA]
  /-
    🎉 no goals
  -/



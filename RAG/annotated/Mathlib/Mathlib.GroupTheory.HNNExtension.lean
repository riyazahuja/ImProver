/-- The relation we quotient the coproduct by to form an `HNNExtension`. -/
def HNNExtension.con (G : Type*) [Group G] (A B : Subgroup G) (φ : A ≃* B) :
    Con (G ∗ Multiplicative ℤ) :=
  conGen (fun x y => ∃ (a : A),
    x = inr (ofAdd 1) * inl (a : G) ∧
    y = inl (φ a : G) * inr (ofAdd 1))


/-- The HNN Extension of a group `G`, `HNNExtension G A B φ`. Given a group `G`, subgroups `A` and
`B` and an isomorphism `φ` of `A` and `B`, we adjoin a letter `t` to `G`, such that for
any `a ∈ A`, the conjugate of `of a` by `t` is `of (φ a)`, where `of` is the canonical
map from `G` into the `HNNExtension`. -/
def HNNExtension (G : Type*) [Group G] (A B : Subgroup G) (φ : A ≃* B) : Type _ :=
  (HNNExtension.con G A B φ).Quotient


instance : Group (HNNExtension G A B φ) := by
  /-
    G : Type u_1
    inst✝² : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    H : Type u_2
    inst✝¹ : Group H
    M : Type u_3
    inst✝ : Monoid M
    ⊢ Group (HNNExtension G A B φ)
  -/
  delta HNNExtension; infer_instance
                      /-
                        🎉 no goals
                      -/


/-- The canonical embedding `G →* HNNExtension G A B φ` -/
def of : G →* HNNExtension G A B φ :=
  (HNNExtension.con G A B φ).mk'.comp inl


/-- The stable letter of the `HNNExtension` -/
def t : HNNExtension G A B φ :=
  (HNNExtension.con G A B φ).mk'.comp inr (ofAdd 1)


theorem t_mul_of (a : A) :
    t * (of (a : G) : HNNExtension G A B φ) = of (φ a : G) * t :=
                                              /-
                                                G : Type u_1
                                                inst✝ : Group G
                                                A B : Subgroup G
                                                φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
                                                a : Subtype fun x => Membership.mem A x
                                                ⊢ And (Eq ((fun x1 x2 => HMul.hMul x1 x2) (Monoid.Coprod.inr (Multiplicative.o …
                                              -/
  (Con.eq _).2 <| ConGen.Rel.of _ _ <| ⟨a, by simp⟩
                                              /-
                                                🎉 no goals
                                              -/


theorem of_mul_t (b : B) :
    (of (b : G) : HNNExtension G A B φ) * t = t * of (φ.symm b : G) := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    b : Subtype fun x => Membership.mem B x
    ⊢ Eq (HMul.hMul (HNNExtension.of ↑b) HNNExtension.t) (HMul.hMul HNNExtension.t …
  -/
  rw [t_mul_of]; simp
                 /-
                   🎉 no goals
                 -/


theorem equiv_eq_conj (a : A) :
    (of (φ a : G) : HNNExtension G A B φ) = t * of (a : G) * t⁻¹ := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    a : Subtype fun x => Membership.mem A x
    ⊢ Eq (HNNExtension.of ↑(φ a)) (HMul.hMul (HMul.hMul HNNExtension.t (HNNExtensi …
  -/
  rw [t_mul_of]; simp
                 /-
                   🎉 no goals
                 -/


theorem equiv_symm_eq_conj (b : B) :
    (of (φ.symm b : G) : HNNExtension G A B φ) = t⁻¹ * of (b : G) * t := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    b : Subtype fun x => Membership.mem B x
    ⊢ Eq (HNNExtension.of ↑(φ.symm b)) (HMul.hMul (HMul.hMul (Inv.inv HNNExtension …
  -/
  rw [mul_assoc, of_mul_t]; simp
                            /-
                              🎉 no goals
                            -/


theorem inv_t_mul_of (b : B) :
    t⁻¹ * (of (b : G) : HNNExtension G A B φ) = of (φ.symm b : G) * t⁻¹ := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    b : Subtype fun x => Membership.mem B x
    ⊢ Eq (HMul.hMul (Inv.inv HNNExtension.t) (HNNExtension.of ↑b)) (HMul.hMul (HNN …
  -/
  rw [equiv_symm_eq_conj]; simp
                           /-
                             🎉 no goals
                           -/


theorem of_mul_inv_t (a : A) :
    (of (a : G) : HNNExtension G A B φ) * t⁻¹ = t⁻¹ * of (φ a : G) := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    a : Subtype fun x => Membership.mem A x
    ⊢ Eq (HMul.hMul (HNNExtension.of ↑a) (Inv.inv HNNExtension.t)) (HMul.hMul (Inv …
  -/
  rw [equiv_eq_conj]; simp [mul_assoc]
                      /-
                        🎉 no goals
                      -/


/-- Define a function `HNNExtension G A B φ →* H`, by defining it on `G` and `t` -/
def lift (f : G →* H) (x : H) (hx : ∀ a : A, x * f ↑a = f (φ a : G) * x) :
    HNNExtension G A B φ →* H :=
  Con.lift _ (Coprod.lift f (zpowersHom H x)) (Con.conGen_le <| by
    /-
      G : Type u_1
      inst✝² : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      H : Type u_2
      inst✝¹ : Group H
      M : Type u_3
      inst✝ : Monoid M
      f : MonoidHom G H
      x : H
      hx : ∀ (a : Subtype fun x => Membership.mem A x), Eq (HMul.hMul x (f ↑a)) (HMu …
      ⊢ ∀ (x_1 y : Monoid.Coprod G (Multiplicative Int)), (Exists fun a => And (Eq x …
    -/
    rintro _ _ ⟨a, rfl, rfl⟩
    /-
      case intro.intro
      G : Type u_1
      inst✝² : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      H : Type u_2
      inst✝¹ : Group H
      M : Type u_3
      inst✝ : Monoid M
      f : MonoidHom G H
      x : H
      hx : ∀ (a : Subtype fun x => Membership.mem A x), Eq (HMul.hMul x (f ↑a)) (HMu …
      a : Subtype fun x => Membership.mem A x
      ⊢ (Con.ker (Monoid.Coprod.lift f ((zpowersHom H) x))) (HMul.hMul (Monoid.Copro …
    -/
    simp [hx])
    /-
      🎉 no goals
    -/


@[simp]
theorem lift_t (f : G →* H) (x : H) (hx : ∀ a : A, x * f ↑a = f (φ a : G) * x) :
    lift f x hx t = x := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    H : Type u_2
    inst✝ : Group H
    f : MonoidHom G H
    x : H
    hx : ∀ (a : Subtype fun x => Membership.mem A x), Eq (HMul.hMul x (f ↑a)) (HMu …
    ⊢ Eq ((HNNExtension.lift f x hx) HNNExtension.t) x
  -/
  delta HNNExtension; simp [lift, t]
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem lift_of (f : G →* H) (x : H) (hx : ∀ a : A, x * f ↑a = f (φ a : G) * x) (g : G) :
    lift f x hx (of g) = f g := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    H : Type u_2
    inst✝ : Group H
    f : MonoidHom G H
    x : H
    hx : ∀ (a : Subtype fun x => Membership.mem A x), Eq (HMul.hMul x (f ↑a)) (HMu …
    g : G
    ⊢ Eq ((HNNExtension.lift f x hx) (HNNExtension.of g)) (f g)
  -/
  delta HNNExtension; simp [lift, of]
                      /-
                        🎉 no goals
                      -/


@[ext high]
theorem hom_ext {f g : HNNExtension G A B φ →* M}
    (hg : f.comp of = g.comp of) (ht : f t = g t) : f = g :=
  (MonoidHom.cancel_right Con.mk'_surjective).mp <|
    Coprod.hom_ext hg (MonoidHom.ext_mint ht)


@[elab_as_elim]
theorem induction_on {motive : HNNExtension G A B φ → Prop}
    (x : HNNExtension G A B φ) (of : ∀ g, motive (of g))
    (t : motive t) (mul : ∀ x y, motive x → motive y → motive (x * y))
    (inv : ∀ x, motive x → motive x⁻¹) : motive x := by
  let S : Subgroup (HNNExtension G A B φ) :=
    { carrier := setOf motive
      one_mem' := by simpa using of 1
      mul_mem' := mul _ _
      inv_mem' := inv _ }
  let f : HNNExtension G A B φ →* S :=
    lift (HNNExtension.of.codRestrict S of)
      ⟨HNNExtension.t, t⟩ (by intro a; ext; simp [equiv_eq_conj, mul_assoc])
  have hf : S.subtype.comp f = MonoidHom.id _ :=
    hom_ext (by ext; simp [f]) (by simp [f])
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    motive : HNNExtension G A B φ → Prop
    x : HNNExtension G A B φ
    of : ∀ (g : G), motive (HNNExtension.of g)
    t : motive HNNExtension.t
    mul : ∀ (x y : HNNExtension G A B φ), motive x → motive y → motive (HMul.hMul  …
    inv : ∀ (x : HNNExtension G A B φ), motive x → motive (Inv.inv x)
    S : Subgroup (HNNExtension G A B φ) := { carrier := setOf motive, mul_mem' :=  …
    f : MonoidHom (HNNExtension G A B φ) (Subtype fun x => Membership.mem S x) :=  …
    hf : Eq (S.subtype.comp f) (MonoidHom.id (HNNExtension G A B φ))
    ⊢ motive x
  -/
  show motive (MonoidHom.id _ x)
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    motive : HNNExtension G A B φ → Prop
    x : HNNExtension G A B φ
    of : ∀ (g : G), motive (HNNExtension.of g)
    t : motive HNNExtension.t
    mul : ∀ (x y : HNNExtension G A B φ), motive x → motive y → motive (HMul.hMul  …
    inv : ∀ (x : HNNExtension G A B φ), motive x → motive (Inv.inv x)
    S : Subgroup (HNNExtension G A B φ) := { carrier := setOf motive, mul_mem' :=  …
    f : MonoidHom (HNNExtension G A B φ) (Subtype fun x => Membership.mem S x) :=  …
    hf : Eq (S.subtype.comp f) (MonoidHom.id (HNNExtension G A B φ))
    ⊢ motive ((MonoidHom.id (HNNExtension G A B φ)) x)
  -/
  rw [← hf]
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    motive : HNNExtension G A B φ → Prop
    x : HNNExtension G A B φ
    of : ∀ (g : G), motive (HNNExtension.of g)
    t : motive HNNExtension.t
    mul : ∀ (x y : HNNExtension G A B φ), motive x → motive y → motive (HMul.hMul  …
    inv : ∀ (x : HNNExtension G A B φ), motive x → motive (Inv.inv x)
    S : Subgroup (HNNExtension G A B φ) := { carrier := setOf motive, mul_mem' :=  …
    f : MonoidHom (HNNExtension G A B φ) (Subtype fun x => Membership.mem S x) :=  …
    hf : Eq (S.subtype.comp f) (MonoidHom.id (HNNExtension G A B φ))
    ⊢ motive ((S.subtype.comp f) x)
  -/
  exact (f x).2
  /-
    🎉 no goals
  -/


/-- To avoid duplicating code, we define `toSubgroup A B u` and `toSubgroupEquiv u`
where `u : ℤˣ` is `1` or `-1`. `toSubgroup A B u` is `A` when `u = 1` and `B` when `u = -1`,
and `toSubgroupEquiv` is `φ` when `u = 1` and `φ⁻¹` when `u = -1`. `toSubgroup u` is the subgroup
such that for any `a ∈ toSubgroup u`, `t ^ (u : ℤ) * a = toSubgroupEquiv a * t ^ (u : ℤ)`. -/
def toSubgroup (u : ℤˣ) : Subgroup G :=
  if u = 1 then A else B


@[simp]
theorem toSubgroup_one : toSubgroup A B 1 = A := rfl


@[simp]
theorem toSubgroup_neg_one : toSubgroup A B (-1) = B := rfl


/-- To avoid duplicating code, we define `toSubgroup A B u` and `toSubgroupEquiv u`
where `u : ℤˣ` is `1` or `-1`. `toSubgroup A B u` is `A` when `u = 1` and `B` when `u = -1`,
and `toSubgroupEquiv` is the group ismorphism from `toSubgroup A B u` to `toSubgroup A B (-u)`.
It is defined to be `φ` when `u = 1` and `φ⁻¹` when `u = -1`. -/
def toSubgroupEquiv (u : ℤˣ) : toSubgroup A B u ≃* toSubgroup A B (-u) :=
  if hu : u = 1 then hu ▸ φ else by
    /-
      G : Type u_1
      inst✝² : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      H : Type u_2
      inst✝¹ : Group H
      M : Type u_3
      inst✝ : Monoid M
      u : Units Int
      hu : Not (Eq u 1)
      ⊢ MulEquiv (Subtype fun x => Membership.mem (HNNExtension.toSubgroup A B u) x) …
    -/
    convert φ.symm <;>
    /-
      case h.e'_1.h.e'_2.h.h.e'_4
      G : Type u_1
      inst✝² : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      H : Type u_2
      inst✝¹ : Group H
      M : Type u_3
      inst✝ : Monoid M
      u : Units Int
      hu : Not (Eq u 1)
      x✝ : G
      ⊢ Eq (HNNExtension.toSubgroup A B u) B
    -/
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
    cases Int.units_eq_one_or u <;> simp_all
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem toSubgroupEquiv_one : toSubgroupEquiv φ 1 = φ := rfl


@[simp]
theorem toSubgroupEquiv_neg_one : toSubgroupEquiv φ (-1) = φ.symm := rfl


@[simp]
theorem toSubgroupEquiv_neg_apply (u : ℤˣ) (a : toSubgroup A B u) :
    (toSubgroupEquiv φ (-u) (toSubgroupEquiv φ u a) : G) = a := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    u : Units Int
    a : Subtype fun x => Membership.mem (HNNExtension.toSubgroup A B u) x
    ⊢ Eq ↑((HNNExtension.toSubgroupEquiv φ (Neg.neg u)) ((HNNExtension.toSubgroupE …
  -/
  rcases Int.units_eq_one_or u with rfl | rfl
  · -- This used to be `simp` before https://github.com/leanprover/lean4/pull/2644
    /-
      case inl
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      a : Subtype fun x => Membership.mem (HNNExtension.toSubgroup A B 1) x
      ⊢ Eq ↑((HNNExtension.toSubgroupEquiv φ (-1)) ((HNNExtension.toSubgroupEquiv φ  …
    -/
    simp; erw [MulEquiv.symm_apply_apply]
          /-
            🎉 no goals
          -/
    /-
      case inr
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      a : Subtype fun x => Membership.mem (HNNExtension.toSubgroup A B (-1)) x
      ⊢ Eq ↑((HNNExtension.toSubgroupEquiv φ (Neg.neg (-1))) ((HNNExtension.toSubgro …
    -/
  · simp only [toSubgroup_neg_one, toSubgroupEquiv_neg_one, SetLike.coe_eq_coe]
    /-
      case inr
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      a : Subtype fun x => Membership.mem (HNNExtension.toSubgroup A B (-1)) x
      ⊢ Eq ((HNNExtension.toSubgroupEquiv φ (Neg.neg (-1))) (φ.symm a)) a
    -/
    exact φ.apply_symm_apply a
    /-
      🎉 no goals
    -/


/-- To put word in the HNN Extension into a normal form, we must choose an element of each right
coset of both `A` and `B`, such that the chosen element of the subgroup itself is `1`. -/
structure TransversalPair : Type _ where
  /-- The transversal of each subgroup -/
  set : ℤˣ → Set G
  /-- We have exactly one element of each coset of the subgroup -/
  compl : ∀ u, IsComplement (toSubgroup A B u : Subgroup G) (set u)


instance TransversalPair.nonempty : Nonempty (TransversalPair G A B) := by
  /-
    G : Type u_1
    inst✝² : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    H : Type u_2
    inst✝¹ : Group H
    M : Type u_3
    inst✝ : Monoid M
    ⊢ Nonempty (HNNExtension.NormalWord.TransversalPair G A B)
  -/
  choose t ht using fun u ↦ (toSubgroup A B u).exists_isComplement_right 1
  /-
    G : Type u_1
    inst✝² : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    H : Type u_2
    inst✝¹ : Group H
    M : Type u_3
    inst✝ : Monoid M
    t : Units Int → Set G
    ht : ∀ (u : Units Int), And (Subgroup.IsComplement (↑(HNNExtension.toSubgroup  …
    ⊢ Nonempty (HNNExtension.NormalWord.TransversalPair G A B)
  -/
  exact ⟨⟨t, fun i ↦ (ht i).1⟩⟩
  /-
    🎉 no goals
  -/


/-- A reduced word is a `head`, which is an element of `G`, followed by the product list of pairs.
There should also be no sequences of the form `t^u * g * t^-u`, where `g` is in
`toSubgroup A B u` This is a less strict condition than required for `NormalWord`. -/
structure ReducedWord : Type _ where
  /-- Every `ReducedWord` is the product of an element of the group and a word made up
  of letters each of which is in the transversal. `head` is that element of the base group. -/
  head : G
  /-- The list of pairs `(ℤˣ × G)`, where each pair `(u, g)` represents the element `t^u * g` of
  `HNNExtension G A B φ` -/
  toList : List (ℤˣ × G)
  /-- There are no sequences of the form `t^u * g * t^-u` where `g ∈ toSubgroup A B u` -/
  chain : toList.Chain' (fun a b => a.2 ∈ toSubgroup A B a.1 → a.1 = b.1)


/-- The empty reduced word. -/
@[simps]
def ReducedWord.empty : ReducedWord G A B :=
  { head := 1
    toList := []
    chain := List.chain'_nil }


/-- The product of a `ReducedWord` as an element of the `HNNExtension` -/
def ReducedWord.prod : ReducedWord G A B → HNNExtension G A B φ :=
  fun w => of w.head * (w.toList.map (fun x => t ^ (x.1 : ℤ) * of x.2)).prod


/-- Given a `TransversalPair`, we can make a normal form for words in the `HNNExtension G A B φ`.
The normal form is a `head`, which is an element of `G`, followed by the product list of pairs,
`t ^ u * g`, where `u` is `1` or `-1` and `g` is the chosen element of its right coset of
`toSubgroup A B u`. There should also be no sequences of the form `t^u * g * t^-u`
where `g ∈ toSubgroup A B u` -/
structure _root_.HNNExtension.NormalWord (d : TransversalPair G A B)
    extends ReducedWord G A B : Type _ where
  /-- Every element `g : G` in the list is the chosen element of its coset -/
  mem_set : ∀ (u : ℤˣ) (g : G), (u, g) ∈ toList → g ∈ d.set u


@[ext]
theorem ext {w w' : NormalWord d}
    (h1 : w.head = w'.head) (h2 : w.toList = w'.toList) : w = w' := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    d : HNNExtension.NormalWord.TransversalPair G A B
    w w' : HNNExtension.NormalWord d
    h1 : Eq w.head w'.head
    h2 : Eq w.toList w'.toList
    ⊢ Eq w w'
  -/
  rcases w with ⟨⟨⟩, _⟩; cases w'; simp_all
                                   /-
                                     🎉 no goals
                                   -/


/-- The empty word -/
@[simps]
def empty : NormalWord d :=
  { head := 1
    toList := []
                  /-
                    G : Type u_1
                    inst✝² : Group G
                    A B : Subgroup G
                    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
                    H : Type u_2
                    inst✝¹ : Group H
                    M : Type u_3
                    inst✝ : Monoid M
                    d : HNNExtension.NormalWord.TransversalPair G A B
                    ⊢ ∀ (u : Units Int) (g : G), Membership.mem { head := 1, toList := List.nil, c …
                  -/
    mem_set := by simp
                  /-
                    🎉 no goals
                  -/
    chain := List.chain'_nil }


/-- The `NormalWord` representing an element `g` of the group `G`, which is just the element `g`
itself. -/
@[simps]
def ofGroup (g : G) : NormalWord d :=
  { head := g
    toList := []
                  /-
                    G : Type u_1
                    inst✝² : Group G
                    A B : Subgroup G
                    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
                    H : Type u_2
                    inst✝¹ : Group H
                    M : Type u_3
                    inst✝ : Monoid M
                    d : HNNExtension.NormalWord.TransversalPair G A B
                    g : G
                    ⊢ ∀ (u : Units Int) (g_1 : G), Membership.mem { head := g, toList := List.nil, …
                  -/
    mem_set := by simp
                  /-
                    🎉 no goals
                  -/
    chain := List.chain'_nil }


instance : Inhabited (NormalWord d) := ⟨empty⟩


instance : MulAction G (NormalWord d) :=
  { smul := fun g w => { w with head := g * w.head }
                   /-
                     G : Type u_1
                     inst✝² : Group G
                     A B : Subgroup G
                     φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
                     H : Type u_2
                     inst✝¹ : Group H
                     M : Type u_3
                     inst✝ : Monoid M
                     d : HNNExtension.NormalWord.TransversalPair G A B
                     ⊢ ∀ (b : HNNExtension.NormalWord d), Eq (HSMul.hSMul 1 b) b
                   -/
    one_smul := by simp [instHSMul]
                   /-
                     🎉 no goals
                   -/
                   /-
                     G : Type u_1
                     inst✝² : Group G
                     A B : Subgroup G
                     φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
                     H : Type u_2
                     inst✝¹ : Group H
                     M : Type u_3
                     inst✝ : Monoid M
                     d : HNNExtension.NormalWord.TransversalPair G A B
                     ⊢ ∀ (x y : G) (b : HNNExtension.NormalWord d), Eq (HSMul.hSMul (HMul.hMul x y) …
                   -/
    mul_smul := by simp [instHSMul, mul_assoc] }
                   /-
                     🎉 no goals
                   -/


theorem group_smul_def (g : G) (w : NormalWord d) :
    g • w = { w with head := g * w.head } := rfl


@[simp]
theorem group_smul_head (g : G) (w : NormalWord d) : (g • w).head = g * w.head := rfl


@[simp]
theorem group_smul_toList (g : G) (w : NormalWord d) : (g • w).toList = w.toList := rfl


                                                /-
                                                  G : Type u_1
                                                  inst✝² : Group G
                                                  A B : Subgroup G
                                                  φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
                                                  H : Type u_2
                                                  inst✝¹ : Group H
                                                  M : Type u_3
                                                  inst✝ : Monoid M
                                                  d : HNNExtension.NormalWord.TransversalPair G A B
                                                  ⊢ ∀ {m₁ m₂ : G}, (∀ (a : HNNExtension.NormalWord d), Eq (HSMul.hSMul m₁ a) (HS …
                                                -/
instance : FaithfulSMul G (NormalWord d) := ⟨by simp [group_smul_def]⟩
                                                /-
                                                  🎉 no goals
                                                -/


/-- A constructor to append an element `g` of `G` and `u : ℤˣ` to a word `w` with sufficient
hypotheses that no normalization or cancellation need take place for the result to be in normal form
-/
@[simps]
def cons (g : G) (u : ℤˣ) (w : NormalWord d) (h1 : w.head ∈ d.set u)
    (h2 : ∀ u' ∈ Option.map Prod.fst w.toList.head?, w.head ∈ toSubgroup A B u → u = u') :
    NormalWord d :=
  { head := g,
    toList := (u, w.head) :: w.toList,
    mem_set := by
      /-
        G : Type u_1
        inst✝² : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        H : Type u_2
        inst✝¹ : Group H
        M : Type u_3
        inst✝ : Monoid M
        d : HNNExtension.NormalWord.TransversalPair G A B
        g : G
        u : Units Int
        w : HNNExtension.NormalWord d
        h1 : Membership.mem (d.set u) w.head
        h2 : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w.toList.head?) u …
        ⊢ ∀ (u_1 : Units Int) (g_1 : G), Membership.mem { head := g, toList := List.co …
      -/
      intro u' g' h'
      /-
        G : Type u_1
        inst✝² : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        H : Type u_2
        inst✝¹ : Group H
        M : Type u_3
        inst✝ : Monoid M
        d : HNNExtension.NormalWord.TransversalPair G A B
        g : G
        u : Units Int
        w : HNNExtension.NormalWord d
        h1 : Membership.mem (d.set u) w.head
        h2 : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w.toList.head?) u …
        u' : Units Int
        g' : G
        h' : Membership.mem { head := g, toList := List.cons { fst := u, snd := w.head …
        ⊢ Membership.mem (d.set u') g'
      -/
      simp only [List.mem_cons, Prod.mk.injEq] at h'
      /-
        G : Type u_1
        inst✝² : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        H : Type u_2
        inst✝¹ : Group H
        M : Type u_3
        inst✝ : Monoid M
        d : HNNExtension.NormalWord.TransversalPair G A B
        g : G
        u : Units Int
        w : HNNExtension.NormalWord d
        h1 : Membership.mem (d.set u) w.head
        h2 : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w.toList.head?) u …
        u' : Units Int
        g' : G
        h' : Or (And (Eq u' u) (Eq g' w.head)) (Membership.mem w.toList { fst := u', s …
        ⊢ Membership.mem (d.set u') g'
      -/
      rcases h' with ⟨rfl, rfl⟩ | h'
      /-
        G : Type u_1
        inst✝² : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        H : Type u_2
        inst✝¹ : Group H
        M : Type u_3
        inst✝ : Monoid M
        d : HNNExtension.NormalWord.TransversalPair G A B
        g : G
        u : Units Int
        w : HNNExtension.NormalWord d
        h1 : Membership.mem (d.set u) w.head
        h2 : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w.toList.head?) u …
        ⊢ List.Chain' (fun a b => Membership.mem (HNNExtension.toSubgroup A B a.1) a.2 …
      -/
        /-
          case inl.intro
          G : Type u_1
          inst✝² : Group G
          A B : Subgroup G
          φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
          H : Type u_2
          inst✝¹ : Group H
          M : Type u_3
          inst✝ : Monoid M
          d : HNNExtension.NormalWord.TransversalPair G A B
          g : G
          w : HNNExtension.NormalWord d
          u' : Units Int
          h1 : Membership.mem (d.set u') w.head
          h2 : ∀ (u'_1 : Units Int), Membership.mem (Option.map Prod.fst w.toList.head?) …
          ⊢ Membership.mem (d.set u') w.head
        -/
      /-
        G : Type u_1
        inst✝² : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        H : Type u_2
        inst✝¹ : Group H
        M : Type u_3
        inst✝ : Monoid M
        d : HNNExtension.NormalWord.TransversalPair G A B
        g : G
        u : Units Int
        w : HNNExtension.NormalWord d
        h1 : Membership.mem (d.set u) w.head
        h2 : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w.toList.head?) u …
        ⊢ ∀ (y : Prod (Units Int) G), Membership.mem w.toList.head? y → Membership.mem …
      -/
      · exact h1
      /-
        case mk
        G : Type u_1
        inst✝² : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        H : Type u_2
        inst✝¹ : Group H
        M : Type u_3
        inst✝ : Monoid M
        d : HNNExtension.NormalWord.TransversalPair G A B
        g : G
        u : Units Int
        w : HNNExtension.NormalWord d
        h1 : Membership.mem (d.set u) w.head
        h2 : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w.toList.head?) u …
        u' : Units Int
        g' : G
        hu' : Membership.mem w.toList.head? { fst := u', snd := g' }
        hw1 : Membership.mem (HNNExtension.toSubgroup A B { fst := u, snd := w.head }. …
        ⊢ Eq { fst := u, snd := w.head }.1 { fst := u', snd := g' }.1
      -/
        /-
          🎉 no goals
        -/
      /-
        🎉 no goals
      -/
        /-
          case inr
          G : Type u_1
          inst✝² : Group G
          A B : Subgroup G
          φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
          H : Type u_2
          inst✝¹ : Group H
          M : Type u_3
          inst✝ : Monoid M
          d : HNNExtension.NormalWord.TransversalPair G A B
          g : G
          u : Units Int
          w : HNNExtension.NormalWord d
          h1 : Membership.mem (d.set u) w.head
          h2 : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w.toList.head?) u …
          u' : Units Int
          g' : G
          h' : Membership.mem w.toList { fst := u', snd := g' }
          ⊢ Membership.mem (d.set u') g'
        -/
      · exact w.mem_set _ _ h'
        /-
          🎉 no goals
        -/
    chain := by
      refine List.chain'_cons'.2 ⟨?_, w.chain⟩
      rintro ⟨u', g'⟩ hu' hw1
      exact h2 _ (by simp_all) hw1 }


/-- A recursor to induct on a `NormalWord`, by proving the property is preserved under `cons` -/
@[elab_as_elim]
def consRecOn {motive : NormalWord d → Sort*} (w : NormalWord d)
    (ofGroup : ∀g, motive (ofGroup g))
    (cons : ∀ (g : G) (u : ℤˣ) (w : NormalWord d) (h1 : w.head ∈ d.set u)
      (h2 : ∀ u' ∈ Option.map Prod.fst w.toList.head?,
        w.head ∈ toSubgroup A B u → u = u'),
      motive w → motive (cons g u w h1 h2)) : motive w := by
  /-
    G : Type u_1
    inst✝² : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    H : Type u_2
    inst✝¹ : Group H
    M : Type u_3
    inst✝ : Monoid M
    d : HNNExtension.NormalWord.TransversalPair G A B
    motive : HNNExtension.NormalWord d → Sort u_4
    w : HNNExtension.NormalWord d
    ofGroup : (g : G) → motive (HNNExtension.NormalWord.ofGroup g)
    cons : (g : G) → (u : Units Int) → (w : HNNExtension.NormalWord d) → (h1 : Mem …
    ⊢ motive w
  -/
  rcases w with ⟨⟨g, l, chain⟩, mem_set⟩
  induction l generalizing g with
  | nil => exact ofGroup _
  | cons a l ih =>
    exact cons g a.1
      { head := a.2
        toList := l
        mem_set := fun _ _ h => mem_set _ _ (List.mem_cons_of_mem _ h),
        chain := (List.chain'_cons'.1 chain).2 }
      (mem_set a.1 a.2 (List.mem_cons_self _ _))
      (by simpa using (List.chain'_cons'.1 chain).1)
      (ih _ _ _)


@[simp]
theorem consRecOn_ofGroup {motive : NormalWord d → Sort*}
    (g : G) (ofGroup : ∀g, motive (ofGroup g))
    (cons : ∀ (g : G) (u : ℤˣ) (w : NormalWord d) (h1 : w.head ∈ d.set u)
      (h2 : ∀ u' ∈ Option.map Prod.fst w.toList.head?, w.head
        ∈ toSubgroup A B u → u = u'),
      motive w → motive (cons g u w h1 h2)) :
    consRecOn (.ofGroup g) ofGroup cons = ofGroup g := rfl


@[simp]
theorem consRecOn_cons {motive : NormalWord d → Sort*}
    (g : G) (u : ℤˣ) (w : NormalWord d) (h1 : w.head ∈ d.set u)
    (h2 : ∀ u' ∈ Option.map Prod.fst w.toList.head?, w.head ∈ toSubgroup A B u → u = u')
    (ofGroup : ∀g, motive (ofGroup g))
    (cons : ∀ (g : G) (u : ℤˣ) (w : NormalWord d) (h1 : w.head ∈ d.set u)
      (h2 : ∀ u' ∈ Option.map Prod.fst w.toList.head?,
        w.head ∈ toSubgroup A B u → u = u'),
      motive w → motive (cons g u w h1 h2)) :
    consRecOn (.cons g u w h1 h2) ofGroup cons = cons g u w h1 h2
      (consRecOn w ofGroup cons) := rfl


@[simp]
theorem smul_cons (g₁ g₂ : G) (u : ℤˣ) (w : NormalWord d) (h1 : w.head ∈ d.set u)
    (h2 : ∀ u' ∈ Option.map Prod.fst w.toList.head?, w.head ∈ toSubgroup A B u → u = u') :
    g₁ • cons g₂ u w h1 h2 = cons (g₁ * g₂) u w h1 h2 :=
  rfl


@[simp]
theorem smul_ofGroup (g₁ g₂ : G) :
    g₁ • (ofGroup g₂ : NormalWord d) = ofGroup (g₁ * g₂) := rfl


/-- The action of `t^u` on `ofGroup g`. The normal form will be
`a * t^u * g'` where `a ∈ toSubgroup A B (-u)` -/
noncomputable def unitsSMulGroup (u : ℤˣ) (g : G) :
    (toSubgroup A B (-u)) × d.set u :=
  let g' := (d.compl u).equiv g
  (toSubgroupEquiv φ u g'.1, g'.2)


theorem unitsSMulGroup_snd (u : ℤˣ) (g : G) :
    (unitsSMulGroup φ d u g).2 = ((d.compl u).equiv g).2 := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    d : HNNExtension.NormalWord.TransversalPair G A B
    u : Units Int
    g : G
    ⊢ Eq (HNNExtension.NormalWord.unitsSMulGroup φ d u g).2 (⋯.equiv g).2
  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  rcases Int.units_eq_one_or u with rfl | rfl <;> rfl
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- `Cancels u w` is a predicate expressing whether `t^u` cancels with some occurrence
of `t^-u` when we multiply `t^u` by `w`. -/
def Cancels (u : ℤˣ) (w : NormalWord d) : Prop :=
  (w.head ∈ (toSubgroup A B u : Subgroup G)) ∧ w.toList.head?.map Prod.fst = some (-u)


/-- Multiplying `t^u` by `w` in the special case where cancellation happens -/
def unitsSMulWithCancel (u : ℤˣ) (w : NormalWord d) : Cancels u w → NormalWord d :=
  consRecOn w
        /-
          G : Type u_1
          inst✝² : Group G
          A B : Subgroup G
          φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
          H : Type u_2
          inst✝¹ : Group H
          M : Type u_3
          inst✝ : Monoid M
          d : HNNExtension.NormalWord.TransversalPair G A B
          u : Units Int
          w : HNNExtension.NormalWord d
          ⊢ (g : G) → HNNExtension.NormalWord.Cancels u (HNNExtension.NormalWord.ofGroup …
        -/
    (by simp [Cancels, ofGroup]; tauto)
                                 /-
                                   🎉 no goals
                                 -/
    (fun g _ w _ _ _ can =>
      (toSubgroupEquiv φ u ⟨g, can.1⟩ : G) • w)


/-- Multiplying `t^u` by a `NormalWord`, `w` and putting the result in normal form. -/
noncomputable def unitsSMul (u : ℤˣ) (w : NormalWord d) : NormalWord d :=
  letI := Classical.dec
  if h : Cancels u w
  then unitsSMulWithCancel φ u w h
  else let g' := unitsSMulGroup φ d u w.head
    cons g'.1 u ((g'.2 * w.head⁻¹ : G) • w)
          /-
            G : Type u_1
            inst✝² : Group G
            A B : Subgroup G
            φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
            H : Type u_2
            inst✝¹ : Group H
            M : Type u_3
            inst✝ : Monoid M
            d : HNNExtension.NormalWord.TransversalPair G A B
            u : Units Int
            w : HNNExtension.NormalWord d
            this : (p : Prop) → Decidable p := Classical.dec
            h : Not (HNNExtension.NormalWord.Cancels u w)
            g' : Prod (Subtype fun x => Membership.mem (HNNExtension.toSubgroup A B (Neg.n …
            ⊢ Membership.mem (d.set u) (HSMul.hSMul (HMul.hMul (↑g'.2) (Inv.inv w.head)) w …
          -/
      (by simp)
          /-
            🎉 no goals
          -/
      (by
        simp only [g', group_smul_toList, Option.mem_def, Option.map_eq_some', Prod.exists,
          exists_and_right, exists_eq_right, group_smul_head, inv_mul_cancel_right,
          forall_exists_index, unitsSMulGroup]
        simp only [Cancels, Option.map_eq_some', Prod.exists, exists_and_right, exists_eq_right,
          not_and, not_exists] at h
        /-
          G : Type u_1
          inst✝² : Group G
          A B : Subgroup G
          φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
          H : Type u_2
          inst✝¹ : Group H
          M : Type u_3
          inst✝ : Monoid M
          d : HNNExtension.NormalWord.TransversalPair G A B
          u : Units Int
          w : HNNExtension.NormalWord d
          this : (p : Prop) → Decidable p := Classical.dec
          g' : Prod (Subtype fun x => Membership.mem (HNNExtension.toSubgroup A B (Neg.n …
          h : Membership.mem (HNNExtension.toSubgroup A B u) w.head → ∀ (x : G), Not (Eq …
          ⊢ ∀ (u' : Units Int) (x : G), Eq w.toList.head? (Option.some { fst := u', snd  …
        -/
        intro u' x hx hmem
        have : w.head ∈ toSubgroup A B u := by
          have := (d.compl u).rightCosetEquivalence_equiv_snd w.head
          rw [RightCosetEquivalence, rightCoset_eq_iff, mul_mem_cancel_left hmem] at this
          simp_all
        /-
          G : Type u_1
          inst✝² : Group G
          A B : Subgroup G
          φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
          H : Type u_2
          inst✝¹ : Group H
          M : Type u_3
          inst✝ : Monoid M
          d : HNNExtension.NormalWord.TransversalPair G A B
          u : Units Int
          w : HNNExtension.NormalWord d
          this✝ : (p : Prop) → Decidable p := Classical.dec
          g' : Prod (Subtype fun x => Membership.mem (HNNExtension.toSubgroup A B (Neg.n …
          h : Membership.mem (HNNExtension.toSubgroup A B u) w.head → ∀ (x : G), Not (Eq …
          u' : Units Int
          x : G
          hx : Eq w.toList.head? (Option.some { fst := u', snd := x })
          hmem : Membership.mem (HNNExtension.toSubgroup A B u) ↑(⋯.equiv w.head).2
          this : Membership.mem (HNNExtension.toSubgroup A B u) w.head
          ⊢ Eq u u'
        -/
        have := h this x
        /-
          G : Type u_1
          inst✝² : Group G
          A B : Subgroup G
          φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
          H : Type u_2
          inst✝¹ : Group H
          M : Type u_3
          inst✝ : Monoid M
          d : HNNExtension.NormalWord.TransversalPair G A B
          u : Units Int
          w : HNNExtension.NormalWord d
          this✝¹ : (p : Prop) → Decidable p := Classical.dec
          g' : Prod (Subtype fun x => Membership.mem (HNNExtension.toSubgroup A B (Neg.n …
          h : Membership.mem (HNNExtension.toSubgroup A B u) w.head → ∀ (x : G), Not (Eq …
          u' : Units Int
          x : G
          hx : Eq w.toList.head? (Option.some { fst := u', snd := x })
          hmem : Membership.mem (HNNExtension.toSubgroup A B u) ↑(⋯.equiv w.head).2
          this✝ : Membership.mem (HNNExtension.toSubgroup A B u) w.head
          this : Not (Eq w.toList.head? (Option.some { fst := Neg.neg u, snd := x }))
          ⊢ Eq u u'
        -/
        simp_all [Int.units_ne_iff_eq_neg])
        /-
          🎉 no goals
        -/


/-- A condition for not cancelling whose hypothese are the same as those of the `cons` function. -/
theorem not_cancels_of_cons_hyp (u : ℤˣ) (w : NormalWord d)
    (h2 : ∀ u' ∈ Option.map Prod.fst w.toList.head?,
      w.head ∈ toSubgroup A B u → u = u') :
    ¬ Cancels u w := by
  simp only [Cancels, Option.map_eq_some', Prod.exists,
    exists_and_right, exists_eq_right, not_and, not_exists]
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    d : HNNExtension.NormalWord.TransversalPair G A B
    u : Units Int
    w : HNNExtension.NormalWord d
    h2 : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w.toList.head?) u …
    ⊢ Membership.mem (HNNExtension.toSubgroup A B u) w.head → ∀ (x : G), Not (Eq w …
  -/
  intro hw x hx
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    d : HNNExtension.NormalWord.TransversalPair G A B
    u : Units Int
    w : HNNExtension.NormalWord d
    h2 : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w.toList.head?) u …
    hw : Membership.mem (HNNExtension.toSubgroup A B u) w.head
    x : G
    hx : Eq w.toList.head? (Option.some { fst := Neg.neg u, snd := x })
    ⊢ False
  -/
  rw [hx] at h2
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    d : HNNExtension.NormalWord.TransversalPair G A B
    u : Units Int
    w : HNNExtension.NormalWord d
    hw : Membership.mem (HNNExtension.toSubgroup A B u) w.head
    x : G
    h2 : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst (Option.some { fs …
    hx : Eq w.toList.head? (Option.some { fst := Neg.neg u, snd := x })
    ⊢ False
  -/
  simpa using h2 (-u) rfl hw
  /-
    🎉 no goals
  -/


theorem unitsSMul_cancels_iff (u : ℤˣ) (w : NormalWord d) :
    Cancels (-u) (unitsSMul φ u w) ↔ ¬ Cancels u w := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    d : HNNExtension.NormalWord.TransversalPair G A B
    u : Units Int
    w : HNNExtension.NormalWord d
    ⊢ Iff (HNNExtension.NormalWord.Cancels (Neg.neg u) (HNNExtension.NormalWord.un …
  -/
  by_cases h : Cancels u w
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      u : Units Int
      w : HNNExtension.NormalWord d
      h : HNNExtension.NormalWord.Cancels u w
      ⊢ Iff (HNNExtension.NormalWord.Cancels (Neg.neg u) (HNNExtension.NormalWord.un …
    -/
  · simp only [unitsSMul, h, dite_true, not_true_eq_false, iff_false]
    induction w using consRecOn with
    | ofGroup => simp [Cancels, unitsSMulWithCancel]
    | cons g u' w h1 h2 _ =>
      intro hc
      apply not_cancels_of_cons_hyp _ _ h2
      simp only [Cancels, cons_head, cons_toList, List.head?_cons,
        Option.map_some', Option.some.injEq] at h
      cases h.2
      simpa [Cancels, unitsSMulWithCancel,
        Subgroup.mul_mem_cancel_left] using hc
    /-
      case neg
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      u : Units Int
      w : HNNExtension.NormalWord d
      h : Not (HNNExtension.NormalWord.Cancels u w)
      ⊢ Iff (HNNExtension.NormalWord.Cancels (Neg.neg u) (HNNExtension.NormalWord.un …
    -/
  · simp only [unitsSMul, dif_neg h]
    /-
      case neg
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      u : Units Int
      w : HNNExtension.NormalWord d
      h : Not (HNNExtension.NormalWord.Cancels u w)
      ⊢ Iff (HNNExtension.NormalWord.Cancels (Neg.neg u) (HNNExtension.NormalWord.co …
    -/
    simpa [Cancels] using h
    /-
      🎉 no goals
    -/


theorem unitsSMul_neg (u : ℤˣ) (w : NormalWord d) :
    unitsSMul φ (-u) (unitsSMul φ u w) = w := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    d : HNNExtension.NormalWord.TransversalPair G A B
    u : Units Int
    w : HNNExtension.NormalWord d
    ⊢ Eq (HNNExtension.NormalWord.unitsSMul φ (Neg.neg u) (HNNExtension.NormalWord …
  -/
  rw [unitsSMul]
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    d : HNNExtension.NormalWord.TransversalPair G A B
    u : Units Int
    w : HNNExtension.NormalWord d
    ⊢ Eq
        (dite (HNNExtension.NormalWord.Cancels (Neg.neg u) (HNNExtension.NormalWor …
          let g' := HNNExtension.NormalWord.unitsSMulGroup φ d (Neg.neg u) (HNNExt …
          HNNExtension.NormalWord.cons (↑g'.1) (Neg.neg u) (HSMul.hSMul (HMul.hMul …
        w
  -/
  split_ifs with hcan
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      u : Units Int
      w : HNNExtension.NormalWord d
      hcan : HNNExtension.NormalWord.Cancels (Neg.neg u) (HNNExtension.NormalWord.un …
      ⊢ Eq (HNNExtension.NormalWord.unitsSMulWithCancel φ (Neg.neg u) (HNNExtension. …
    -/
  · have hncan : ¬ Cancels u w := (unitsSMul_cancels_iff _ _ _).1 hcan
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      u : Units Int
      w : HNNExtension.NormalWord d
      hcan : HNNExtension.NormalWord.Cancels (Neg.neg u) (HNNExtension.NormalWord.un …
      hncan : Not (HNNExtension.NormalWord.Cancels u w)
      ⊢ Eq (HNNExtension.NormalWord.unitsSMulWithCancel φ (Neg.neg u) (HNNExtension. …
    -/
    unfold unitsSMul
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      u : Units Int
      w : HNNExtension.NormalWord d
      hcan : HNNExtension.NormalWord.Cancels (Neg.neg u) (HNNExtension.NormalWord.un …
      hncan : Not (HNNExtension.NormalWord.Cancels u w)
      ⊢ Eq
          (HNNExtension.NormalWord.unitsSMulWithCancel φ (Neg.neg u)
            (dite (HNNExtension.NormalWord.Cancels u w) (fun h => HNNExtension.Norma …
              let g' := HNNExtension.NormalWord.unitsSMulGroup φ d u w.head;
              HNNExtension.NormalWord.cons (↑g'.1) u (HSMul.hSMul (HMul.hMul (↑g'.2) …
            hcan)
          w
    -/
    simp only [dif_neg hncan]
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      u : Units Int
      w : HNNExtension.NormalWord d
      hcan : HNNExtension.NormalWord.Cancels (Neg.neg u) (HNNExtension.NormalWord.un …
      hncan : Not (HNNExtension.NormalWord.Cancels u w)
      ⊢ Eq (HNNExtension.NormalWord.unitsSMulWithCancel φ (Neg.neg u) (HNNExtension. …
    -/
    simp [unitsSMulWithCancel, unitsSMulGroup, (d.compl u).equiv_snd_eq_inv_mul]
    -- This used to be the end of the proof before https://github.com/leanprover/lean4/pull/2644
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      u : Units Int
      w : HNNExtension.NormalWord d
      hcan : HNNExtension.NormalWord.Cancels (Neg.neg u) (HNNExtension.NormalWord.un …
      hncan : Not (HNNExtension.NormalWord.Cancels u w)
      ⊢ Eq (HSMul.hSMul (↑(⋯.equiv w.head).1) (HSMul.hSMul (HMul.hMul (↑(⋯.equiv w.h …
    -/
    erw [(d.compl u).equiv_snd_eq_inv_mul]
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      u : Units Int
      w : HNNExtension.NormalWord d
      hcan : HNNExtension.NormalWord.Cancels (Neg.neg u) (HNNExtension.NormalWord.un …
      hncan : Not (HNNExtension.NormalWord.Cancels u w)
      ⊢ Eq (HSMul.hSMul (↑(⋯.equiv w.head).1) (HSMul.hSMul (HMul.hMul (HMul.hMul (In …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      u : Units Int
      w : HNNExtension.NormalWord d
      hcan : Not (HNNExtension.NormalWord.Cancels (Neg.neg u) (HNNExtension.NormalWo …
      ⊢ Eq
          (let g' := HNNExtension.NormalWord.unitsSMulGroup φ d (Neg.neg u) (HNNExte …
          HNNExtension.NormalWord.cons (↑g'.1) (Neg.neg u) (HSMul.hSMul (HMul.hMul ( …
          w
    -/
  · have hcan2 : Cancels u w := not_not.1 (mt (unitsSMul_cancels_iff _ _ _).2 hcan)
    /-
      case neg
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      u : Units Int
      w : HNNExtension.NormalWord d
      hcan : Not (HNNExtension.NormalWord.Cancels (Neg.neg u) (HNNExtension.NormalWo …
      hcan2 : HNNExtension.NormalWord.Cancels u w
      ⊢ Eq
          (let g' := HNNExtension.NormalWord.unitsSMulGroup φ d (Neg.neg u) (HNNExte …
          HNNExtension.NormalWord.cons (↑g'.1) (Neg.neg u) (HSMul.hSMul (HMul.hMul ( …
          w
    -/
    unfold unitsSMul at hcan ⊢
    /-
      case neg
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      u : Units Int
      w : HNNExtension.NormalWord d
      hcan :
        Not
          (HNNExtension.NormalWord.Cancels (Neg.neg u)
            (dite (HNNExtension.NormalWord.Cancels u w) (fun h => HNNExtension.Norma …
              let g' := HNNExtension.NormalWord.unitsSMulGroup φ d u w.head;
              HNNExtension.NormalWord.cons (↑g'.1) u (HSMul.hSMul (HMul.hMul (↑g'.2) …
      hcan2 : HNNExtension.NormalWord.Cancels u w
      ⊢ Eq
          (let g' :=
            HNNExtension.NormalWord.unitsSMulGroup φ d (Neg.neg u)
              (dite (HNNExtension.NormalWord.Cancels u w) (fun h => HNNExtension.Nor …
                  let g' := HNNExtension.NormalWord.unitsSMulGroup φ d u w.head;
                  HNNExtension.NormalWord.cons (↑g'.1) u (HSMul.hSMul (HMul.hMul (↑g …
          HNNExtension.NormalWord.cons (↑g'.1) (Neg.neg u)
            (HSMul.hSMul
              (HMul.hMul (↑g'.2)
                (Inv.inv
                  (dite (HNNExtension.NormalWord.Cancels u w) (fun h => HNNExtension …
                      let g' := HNNExtension.NormalWord.unitsSMulGroup φ d u w.head;
                      HNNExtension.NormalWord.cons (↑g'.1) u (HSMul.hSMul (HMul.hMul …
              (dite (HNNExtension.NormalWord.Cancels u w) (fun h => HNNExtension.Nor …
                let g' := HNNExtension.NormalWord.unitsSMulGroup φ d u w.head;
                HNNExtension.NormalWord.cons (↑g'.1) u (HSMul.hSMul (HMul.hMul (↑g'. …
            ⋯ ⋯)
          w
    -/
    simp only [dif_pos hcan2] at hcan ⊢
    cases w using consRecOn with
    | ofGroup => simp [Cancels] at hcan2
    | cons g u' w h1 h2 ih =>
      clear ih
      simp only [unitsSMulGroup, SetLike.coe_sort_coe, unitsSMulWithCancel, id_eq, consRecOn_cons,
        group_smul_head, IsComplement.equiv_mul_left, map_mul, Submonoid.coe_mul, coe_toSubmonoid,
        toSubgroupEquiv_neg_apply, mul_inv_rev]
      cases hcan2.2
      have : ((d.compl (-u)).equiv w.head).1 = 1 :=
        (d.compl (-u)).equiv_fst_eq_one_of_mem_of_one_mem _ h1
      apply NormalWord.ext
      · -- This used to `simp [this]` before https://github.com/leanprover/lean4/pull/2644
        dsimp
        conv_lhs => erw [IsComplement.equiv_mul_left]
        rw [map_mul, Submonoid.coe_mul, toSubgroupEquiv_neg_apply, this]
        simp
      · -- The next two lines were not needed before https://github.com/leanprover/lean4/pull/2644
        dsimp
        conv_lhs => erw [IsComplement.equiv_mul_left]
        simp [mul_assoc, Units.ext_iff, (d.compl (-u)).equiv_snd_eq_inv_mul, this]
        -- The next two lines were not needed before https://github.com/leanprover/lean4/pull/2644
        erw [(d.compl (-u)).equiv_snd_eq_inv_mul, this]
        simp


/-- the equivalence given by multiplication on the left by `t`  -/
@[simps]
noncomputable def unitsSMulEquiv : NormalWord d ≃ NormalWord d :=
  { toFun := unitsSMul φ 1
    invFun := unitsSMul φ (-1),
                            /-
                              G : Type u_1
                              inst✝² : Group G
                              A B : Subgroup G
                              φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
                              H : Type u_2
                              inst✝¹ : Group H
                              M : Type u_3
                              inst✝ : Monoid M
                              d : HNNExtension.NormalWord.TransversalPair G A B
                              x✝ : HNNExtension.NormalWord d
                              ⊢ Eq (HNNExtension.NormalWord.unitsSMul φ (-1) (HNNExtension.NormalWord.unitsS …
                            -/
    left_inv := fun _ => by rw [unitsSMul_neg]
                            /-
                              🎉 no goals
                            -/
                             /-
                               G : Type u_1
                               inst✝² : Group G
                               A B : Subgroup G
                               φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
                               H : Type u_2
                               inst✝¹ : Group H
                               M : Type u_3
                               inst✝ : Monoid M
                               d : HNNExtension.NormalWord.TransversalPair G A B
                               w : HNNExtension.NormalWord d
                               ⊢ Eq (HNNExtension.NormalWord.unitsSMul φ 1 (HNNExtension.NormalWord.unitsSMul …
                             -/
    right_inv := fun w => by convert unitsSMul_neg _ _ w; simp }
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem unitsSMul_one_group_smul (g : A) (w : NormalWord d) :
    unitsSMul φ 1 ((g : G) • w) = (φ g : G) • (unitsSMul φ 1 w) := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    d : HNNExtension.NormalWord.TransversalPair G A B
    g : Subtype fun x => Membership.mem A x
    w : HNNExtension.NormalWord d
    ⊢ Eq (HNNExtension.NormalWord.unitsSMul φ 1 (HSMul.hSMul (↑g) w)) (HSMul.hSMul …
  -/
  unfold unitsSMul
  have : Cancels 1 ((g : G) • w) ↔ Cancels 1 w := by
    simp [Cancels, Subgroup.mul_mem_cancel_left]
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    d : HNNExtension.NormalWord.TransversalPair G A B
    g : Subtype fun x => Membership.mem A x
    w : HNNExtension.NormalWord d
    this : Iff (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) w)) (HNNExtens …
    ⊢ Eq
        (dite (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) w)) (fun h => H …
          let g' := HNNExtension.NormalWord.unitsSMulGroup φ d 1 (HSMul.hSMul (↑g) …
          HNNExtension.NormalWord.cons (↑g'.1) 1 (HSMul.hSMul (HMul.hMul (↑g'.2) ( …
        (HSMul.hSMul (↑(φ g))
          (dite (HNNExtension.NormalWord.Cancels 1 w) (fun h => HNNExtension.Norma …
            let g' := HNNExtension.NormalWord.unitsSMulGroup φ d 1 w.head;
            HNNExtension.NormalWord.cons (↑g'.1) 1 (HSMul.hSMul (HMul.hMul (↑g'.2) …
  -/
  by_cases hcan : Cancels 1 w
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      g : Subtype fun x => Membership.mem A x
      w : HNNExtension.NormalWord d
      this : Iff (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) w)) (HNNExtens …
      hcan : HNNExtension.NormalWord.Cancels 1 w
      ⊢ Eq
          (dite (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) w)) (fun h => H …
            let g' := HNNExtension.NormalWord.unitsSMulGroup φ d 1 (HSMul.hSMul (↑g) …
            HNNExtension.NormalWord.cons (↑g'.1) 1 (HSMul.hSMul (HMul.hMul (↑g'.2) ( …
          (HSMul.hSMul (↑(φ g))
            (dite (HNNExtension.NormalWord.Cancels 1 w) (fun h => HNNExtension.Norma …
              let g' := HNNExtension.NormalWord.unitsSMulGroup φ d 1 w.head;
              HNNExtension.NormalWord.cons (↑g'.1) 1 (HSMul.hSMul (HMul.hMul (↑g'.2) …
    -/
  · simp [unitsSMulWithCancel, dif_pos (this.2 hcan), dif_pos hcan]
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      g : Subtype fun x => Membership.mem A x
      w : HNNExtension.NormalWord d
      this : Iff (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) w)) (HNNExtens …
      hcan : HNNExtension.NormalWord.Cancels 1 w
      ⊢ Eq (HNNExtension.NormalWord.consRecOn (motive := fun x => HNNExtension.Norma …
    -/
    cases w using consRecOn
      /-
        case pos.ofGroup
        G : Type u_1
        inst✝ : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        d : HNNExtension.NormalWord.TransversalPair G A B
        g : Subtype fun x => Membership.mem A x
        g✝ : G
        this : Iff (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) (HNNExtension. …
        hcan : HNNExtension.NormalWord.Cancels 1 (HNNExtension.NormalWord.ofGroup g✝)
        ⊢ Eq (HNNExtension.NormalWord.consRecOn (motive := fun x => HNNExtension.Norma …
      -/
    · simp [Cancels] at hcan
      /-
        🎉 no goals
      -/
      /-
        case pos.cons
        G : Type u_1
        inst✝ : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        d : HNNExtension.NormalWord.TransversalPair G A B
        g : Subtype fun x => Membership.mem A x
        g✝ : G
        u✝ : Units Int
        w✝ : HNNExtension.NormalWord d
        h1✝ : Membership.mem (d.set u✝) w✝.head
        h2✝ : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w✝.toList.head?) …
        this : Iff (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) (HNNExtension. …
        hcan : HNNExtension.NormalWord.Cancels 1 (HNNExtension.NormalWord.cons g✝ u✝ w …
        a✝ : Eq (HNNExtension.NormalWord.cons g✝ u✝ w✝ h1✝ h2✝) w✝ → Eq (HNNExtension. …
        ⊢ Eq (HNNExtension.NormalWord.consRecOn (motive := fun x => HNNExtension.Norma …
      -/
    · simp only [smul_cons, consRecOn_cons, mul_smul]
      /-
        case pos.cons
        G : Type u_1
        inst✝ : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        d : HNNExtension.NormalWord.TransversalPair G A B
        g : Subtype fun x => Membership.mem A x
        g✝ : G
        u✝ : Units Int
        w✝ : HNNExtension.NormalWord d
        h1✝ : Membership.mem (d.set u✝) w✝.head
        h2✝ : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w✝.toList.head?) …
        this : Iff (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) (HNNExtension. …
        hcan : HNNExtension.NormalWord.Cancels 1 (HNNExtension.NormalWord.cons g✝ u✝ w …
        a✝ : Eq (HNNExtension.NormalWord.cons g✝ u✝ w✝ h1✝ h2✝) w✝ → Eq (HNNExtension. …
        ⊢ Eq (HSMul.hSMul (↑(φ ⟨HMul.hMul (↑g) g✝, ⋯⟩)) w✝) (HSMul.hSMul (↑(φ g)) (HSM …
      -/
      rw [← mul_smul, ← Subgroup.coe_mul, ← map_mul φ]
      /-
        case pos.cons
        G : Type u_1
        inst✝ : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        d : HNNExtension.NormalWord.TransversalPair G A B
        g : Subtype fun x => Membership.mem A x
        g✝ : G
        u✝ : Units Int
        w✝ : HNNExtension.NormalWord d
        h1✝ : Membership.mem (d.set u✝) w✝.head
        h2✝ : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w✝.toList.head?) …
        this : Iff (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) (HNNExtension. …
        hcan : HNNExtension.NormalWord.Cancels 1 (HNNExtension.NormalWord.cons g✝ u✝ w …
        a✝ : Eq (HNNExtension.NormalWord.cons g✝ u✝ w✝ h1✝ h2✝) w✝ → Eq (HNNExtension. …
        ⊢ Eq (HSMul.hSMul (↑(φ ⟨HMul.hMul (↑g) g✝, ⋯⟩)) w✝) (HSMul.hSMul (↑(φ (HMul.hM …
      -/
      rfl
      /-
        🎉 no goals
      -/
    /-
      case neg
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      g : Subtype fun x => Membership.mem A x
      w : HNNExtension.NormalWord d
      this : Iff (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) w)) (HNNExtens …
      hcan : Not (HNNExtension.NormalWord.Cancels 1 w)
      ⊢ Eq
          (dite (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) w)) (fun h => H …
            let g' := HNNExtension.NormalWord.unitsSMulGroup φ d 1 (HSMul.hSMul (↑g) …
            HNNExtension.NormalWord.cons (↑g'.1) 1 (HSMul.hSMul (HMul.hMul (↑g'.2) ( …
          (HSMul.hSMul (↑(φ g))
            (dite (HNNExtension.NormalWord.Cancels 1 w) (fun h => HNNExtension.Norma …
              let g' := HNNExtension.NormalWord.unitsSMulGroup φ d 1 w.head;
              HNNExtension.NormalWord.cons (↑g'.1) 1 (HSMul.hSMul (HMul.hMul (↑g'.2) …
    -/
  · rw [dif_neg (mt this.1 hcan), dif_neg hcan]
    /-
      case neg
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      g : Subtype fun x => Membership.mem A x
      w : HNNExtension.NormalWord d
      this : Iff (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) w)) (HNNExtens …
      hcan : Not (HNNExtension.NormalWord.Cancels 1 w)
      ⊢ Eq
          (let g' := HNNExtension.NormalWord.unitsSMulGroup φ d 1 (HSMul.hSMul (↑g)  …
          HNNExtension.NormalWord.cons (↑g'.1) 1 (HSMul.hSMul (HMul.hMul (↑g'.2) (In …
          (HSMul.hSMul (↑(φ g))
            (let g' := HNNExtension.NormalWord.unitsSMulGroup φ d 1 w.head;
            HNNExtension.NormalWord.cons (↑g'.1) 1 (HSMul.hSMul (HMul.hMul (↑g'.2) ( …
    -/
    simp [← mul_smul, mul_assoc, unitsSMulGroup]
    -- This used to be the end of the proof before https://github.com/leanprover/lean4/pull/2644
    /-
      case neg
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      g : Subtype fun x => Membership.mem A x
      w : HNNExtension.NormalWord d
      this : Iff (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) w)) (HNNExtens …
      hcan : Not (HNNExtension.NormalWord.Cancels 1 w)
      ⊢ Eq (HNNExtension.NormalWord.cons (↑(φ (⋯.equiv (HMul.hMul (↑g) w.head)).1))  …
    -/
    dsimp
    /-
      case neg
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      g : Subtype fun x => Membership.mem A x
      w : HNNExtension.NormalWord d
      this : Iff (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) w)) (HNNExtens …
      hcan : Not (HNNExtension.NormalWord.Cancels 1 w)
      ⊢ Eq (HNNExtension.NormalWord.cons (↑(φ (⋯.equiv (HMul.hMul (↑g) w.head)).1))  …
    -/
    congr 1
      /-
        case neg.e_g
        G : Type u_1
        inst✝ : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        d : HNNExtension.NormalWord.TransversalPair G A B
        g : Subtype fun x => Membership.mem A x
        w : HNNExtension.NormalWord d
        this : Iff (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) w)) (HNNExtens …
        hcan : Not (HNNExtension.NormalWord.Cancels 1 w)
        ⊢ Eq (↑(φ (⋯.equiv (HMul.hMul (↑g) w.head)).1)) (HMul.hMul ↑(φ g) ↑(φ (⋯.equiv …
      -/
    · conv_lhs => erw [IsComplement.equiv_mul_left]
      /-
        case neg.e_g
        G : Type u_1
        inst✝ : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        d : HNNExtension.NormalWord.TransversalPair G A B
        g : Subtype fun x => Membership.mem A x
        w : HNNExtension.NormalWord d
        this : Iff (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) w)) (HNNExtens …
        hcan : Not (HNNExtension.NormalWord.Cancels 1 w)
        ⊢ Eq (↑(φ { fst := HMul.hMul g (⋯.equiv w.head).1, snd := (⋯.equiv w.head).2 } …
      -/
      simp_rw [toSubgroup_one]
      /-
        case neg.e_g
        G : Type u_1
        inst✝ : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        d : HNNExtension.NormalWord.TransversalPair G A B
        g : Subtype fun x => Membership.mem A x
        w : HNNExtension.NormalWord d
        this : Iff (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) w)) (HNNExtens …
        hcan : Not (HNNExtension.NormalWord.Cancels 1 w)
        ⊢ Eq (↑(φ (HMul.hMul g (⋯.equiv w.head).1))) (HMul.hMul ↑(φ g) ↑(φ (⋯.equiv w. …
      -/
      simp only [SetLike.coe_sort_coe, map_mul, Subgroup.coe_mul]
      /-
        🎉 no goals
      -/
    /-
      case neg.e_w
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      g : Subtype fun x => Membership.mem A x
      w : HNNExtension.NormalWord d
      this : Iff (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) w)) (HNNExtens …
      hcan : Not (HNNExtension.NormalWord.Cancels 1 w)
      ⊢ Eq (HSMul.hSMul (HMul.hMul (↑(⋯.equiv (HMul.hMul (↑g) w.head)).2) (Inv.inv w …
    -/
    conv_lhs => erw [IsComplement.equiv_mul_left]
    /-
      case neg.e_w
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      g : Subtype fun x => Membership.mem A x
      w : HNNExtension.NormalWord d
      this : Iff (HNNExtension.NormalWord.Cancels 1 (HSMul.hSMul (↑g) w)) (HNNExtens …
      hcan : Not (HNNExtension.NormalWord.Cancels 1 w)
      ⊢ Eq (HSMul.hSMul (HMul.hMul (↑{ fst := HMul.hMul g (⋯.equiv w.head).1, snd := …
    -/
    rfl
    /-
      🎉 no goals
    -/


noncomputable instance : MulAction (HNNExtension G A B φ) (NormalWord d) :=
  MulAction.ofEndHom <| (MulAction.toEndHom (M := Equiv.Perm (NormalWord d))).comp
    (HNNExtension.lift (MulAction.toPermHom _ _) (unitsSMulEquiv φ) <| by
      /-
        G : Type u_1
        inst✝² : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        H : Type u_2
        inst✝¹ : Group H
        M : Type u_3
        inst✝ : Monoid M
        d : HNNExtension.NormalWord.TransversalPair G A B
        ⊢ ∀ (a : Subtype fun x => Membership.mem A x), Eq (HMul.hMul (HNNExtension.Nor …
      -/
      intro a
      /-
        G : Type u_1
        inst✝² : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        H : Type u_2
        inst✝¹ : Group H
        M : Type u_3
        inst✝ : Monoid M
        d : HNNExtension.NormalWord.TransversalPair G A B
        a : Subtype fun x => Membership.mem A x
        ⊢ Eq (HMul.hMul (HNNExtension.NormalWord.unitsSMulEquiv φ) ((MulAction.toPermH …
      -/
      ext : 1
      /-
        case H
        G : Type u_1
        inst✝² : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        H : Type u_2
        inst✝¹ : Group H
        M : Type u_3
        inst✝ : Monoid M
        d : HNNExtension.NormalWord.TransversalPair G A B
        a : Subtype fun x => Membership.mem A x
        x✝ : HNNExtension.NormalWord d
        ⊢ Eq ((HMul.hMul (HNNExtension.NormalWord.unitsSMulEquiv φ) ((MulAction.toPerm …
      -/
      simp [unitsSMul_one_group_smul])
      /-
        🎉 no goals
      -/


@[simp]
theorem prod_group_smul (g : G) (w : NormalWord d) :
    (g • w).prod φ = of g * (w.prod φ) := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    d : HNNExtension.NormalWord.TransversalPair G A B
    g : G
    w : HNNExtension.NormalWord d
    ⊢ Eq (HNNExtension.NormalWord.ReducedWord.prod φ (HSMul.hSMul g w).toReducedWo …
  -/
  simp [ReducedWord.prod, smul_def, mul_assoc]
  /-
    🎉 no goals
  -/


theorem of_smul_eq_smul (g : G) (w : NormalWord d) :
    (of g : HNNExtension G A B φ) • w = g • w := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    d : HNNExtension.NormalWord.TransversalPair G A B
    g : G
    w : HNNExtension.NormalWord d
    ⊢ Eq (HSMul.hSMul (HNNExtension.of g) w) (HSMul.hSMul g w)
  -/
  simp [instHSMul, SMul.smul, MulAction.toEndHom]
  /-
    🎉 no goals
  -/


theorem t_smul_eq_unitsSMul (w : NormalWord d) :
    (t : HNNExtension G A B φ) • w = unitsSMul φ 1 w := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    d : HNNExtension.NormalWord.TransversalPair G A B
    w : HNNExtension.NormalWord d
    ⊢ Eq (HSMul.hSMul HNNExtension.t w) (HNNExtension.NormalWord.unitsSMul φ 1 w)
  -/
  simp [instHSMul, SMul.smul, MulAction.toEndHom]
  /-
    🎉 no goals
  -/


theorem t_pow_smul_eq_unitsSMul (u : ℤˣ) (w : NormalWord d) :
    (t ^ (u : ℤ) : HNNExtension G A B φ) • w = unitsSMul φ u w := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    d : HNNExtension.NormalWord.TransversalPair G A B
    u : Units Int
    w : HNNExtension.NormalWord d
    ⊢ Eq (HSMul.hSMul (HPow.hPow HNNExtension.t ↑u) w) (HNNExtension.NormalWord.un …
  -/
  rcases Int.units_eq_one_or u with (rfl | rfl) <;>
    /-
      case inl
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      w : HNNExtension.NormalWord d
      ⊢ Eq (HSMul.hSMul (HPow.hPow HNNExtension.t ↑1) w) (HNNExtension.NormalWord.un …
    -/
    /-
      🎉 no goals
    -/
    simp [instHSMul, SMul.smul, MulAction.toEndHom, Equiv.Perm.inv_def]
    /-
      🎉 no goals
    -/


@[simp]
theorem prod_cons (g : G) (u : ℤˣ) (w : NormalWord d) (h1 : w.head ∈ d.set u)
    (h2 : ∀ u' ∈ Option.map Prod.fst w.toList.head?,
      w.head ∈ toSubgroup A B u → u = u') :
    (cons g u w h1 h2).prod φ = of g * (t ^ (u : ℤ) * w.prod φ) := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    d : HNNExtension.NormalWord.TransversalPair G A B
    g : G
    u : Units Int
    w : HNNExtension.NormalWord d
    h1 : Membership.mem (d.set u) w.head
    h2 : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w.toList.head?) u …
    ⊢ Eq (HNNExtension.NormalWord.ReducedWord.prod φ (HNNExtension.NormalWord.cons …
  -/
  simp [ReducedWord.prod, cons, smul_def, mul_assoc]
  /-
    🎉 no goals
  -/


theorem prod_unitsSMul (u : ℤˣ) (w : NormalWord d) :
    (unitsSMul φ u w).prod φ = (t^(u : ℤ) * w.prod φ : HNNExtension G A B φ) := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    d : HNNExtension.NormalWord.TransversalPair G A B
    u : Units Int
    w : HNNExtension.NormalWord d
    ⊢ Eq (HNNExtension.NormalWord.ReducedWord.prod φ (HNNExtension.NormalWord.unit …
  -/
  rw [unitsSMul]
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    d : HNNExtension.NormalWord.TransversalPair G A B
    u : Units Int
    w : HNNExtension.NormalWord d
    ⊢ Eq
        (HNNExtension.NormalWord.ReducedWord.prod φ
          (dite (HNNExtension.NormalWord.Cancels u w) (fun h => HNNExtension.Norma …
              let g' := HNNExtension.NormalWord.unitsSMulGroup φ d u w.head;
              HNNExtension.NormalWord.cons (↑g'.1) u (HSMul.hSMul (HMul.hMul (↑g'. …
        (HMul.hMul (HPow.hPow HNNExtension.t ↑u) (HNNExtension.NormalWord.ReducedW …
  -/
  split_ifs with hcan
    /-
      case pos
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      u : Units Int
      w : HNNExtension.NormalWord d
      hcan : HNNExtension.NormalWord.Cancels u w
      ⊢ Eq (HNNExtension.NormalWord.ReducedWord.prod φ (HNNExtension.NormalWord.unit …
    -/
  · cases w using consRecOn
      /-
        case pos.ofGroup
        G : Type u_1
        inst✝ : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        d : HNNExtension.NormalWord.TransversalPair G A B
        u : Units Int
        g✝ : G
        hcan : HNNExtension.NormalWord.Cancels u (HNNExtension.NormalWord.ofGroup g✝)
        ⊢ Eq (HNNExtension.NormalWord.ReducedWord.prod φ (HNNExtension.NormalWord.unit …
      -/
    · simp [Cancels] at hcan
      /-
        🎉 no goals
      -/
      /-
        case pos.cons
        G : Type u_1
        inst✝ : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        d : HNNExtension.NormalWord.TransversalPair G A B
        u : Units Int
        g✝ : G
        u✝ : Units Int
        w✝ : HNNExtension.NormalWord d
        h1✝ : Membership.mem (d.set u✝) w✝.head
        h2✝ : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w✝.toList.head?) …
        hcan : HNNExtension.NormalWord.Cancels u (HNNExtension.NormalWord.cons g✝ u✝ w …
        a✝ : Eq (HNNExtension.NormalWord.cons g✝ u✝ w✝ h1✝ h2✝) w✝ → Eq (HNNExtension. …
        ⊢ Eq (HNNExtension.NormalWord.ReducedWord.prod φ (HNNExtension.NormalWord.unit …
      -/
    · cases hcan.2
      /-
        case pos.cons.refl
        G : Type u_1
        inst✝ : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        d : HNNExtension.NormalWord.TransversalPair G A B
        u : Units Int
        g✝ : G
        w✝ : HNNExtension.NormalWord d
        h1✝ : Membership.mem (d.set { val := Neg.neg ↑u, inv := Neg.neg ↑(Inv.inv u),  …
        h2✝ : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w✝.toList.head?) …
        hcan : HNNExtension.NormalWord.Cancels u (HNNExtension.NormalWord.cons g✝ { va …
        a✝ : Eq (HNNExtension.NormalWord.cons g✝ { val := Neg.neg ↑u, inv := Neg.neg ↑ …
        ⊢ Eq (HNNExtension.NormalWord.ReducedWord.prod φ (HNNExtension.NormalWord.unit …
      -/
      simp [unitsSMulWithCancel]
      /-
        case pos.cons.refl
        G : Type u_1
        inst✝ : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        d : HNNExtension.NormalWord.TransversalPair G A B
        u : Units Int
        g✝ : G
        w✝ : HNNExtension.NormalWord d
        h1✝ : Membership.mem (d.set { val := Neg.neg ↑u, inv := Neg.neg ↑(Inv.inv u),  …
        h2✝ : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w✝.toList.head?) …
        hcan : HNNExtension.NormalWord.Cancels u (HNNExtension.NormalWord.cons g✝ { va …
        a✝ : Eq (HNNExtension.NormalWord.cons g✝ { val := Neg.neg ↑u, inv := Neg.neg ↑ …
        ⊢ Eq (HMul.hMul (HNNExtension.of ↑((HNNExtension.toSubgroupEquiv φ u) ⟨g✝, ⋯⟩) …
      -/
      rcases Int.units_eq_one_or u with (rfl | rfl)
        /-
          case pos.cons.refl.inl
          G : Type u_1
          inst✝ : Group G
          A B : Subgroup G
          φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
          d : HNNExtension.NormalWord.TransversalPair G A B
          g✝ : G
          w✝ : HNNExtension.NormalWord d
          h1✝ : Membership.mem (d.set { val := Neg.neg ↑1, inv := Neg.neg ↑(Inv.inv 1),  …
          h2✝ : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w✝.toList.head?) …
          hcan : HNNExtension.NormalWord.Cancels 1 (HNNExtension.NormalWord.cons g✝ { va …
          a✝ : Eq (HNNExtension.NormalWord.cons g✝ { val := Neg.neg ↑1, inv := Neg.neg ↑ …
          ⊢ Eq (HMul.hMul (HNNExtension.of ↑((HNNExtension.toSubgroupEquiv φ 1) ⟨g✝, ⋯⟩) …
        -/
      · simp [equiv_eq_conj, mul_assoc]
        /-
          🎉 no goals
        -/
        /-
          case pos.cons.refl.inr
          G : Type u_1
          inst✝ : Group G
          A B : Subgroup G
          φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
          d : HNNExtension.NormalWord.TransversalPair G A B
          g✝ : G
          w✝ : HNNExtension.NormalWord d
          h1✝ : Membership.mem (d.set { val := Neg.neg ↑(-1), inv := Neg.neg ↑(Inv.inv ( …
          h2✝ : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w✝.toList.head?) …
          hcan : HNNExtension.NormalWord.Cancels (-1) (HNNExtension.NormalWord.cons g✝ { …
          a✝ : Eq (HNNExtension.NormalWord.cons g✝ { val := Neg.neg ↑(-1), inv := Neg.ne …
          ⊢ Eq (HMul.hMul (HNNExtension.of ↑((HNNExtension.toSubgroupEquiv φ (-1)) ⟨g✝,  …
        -/
      · simp [equiv_symm_eq_conj, mul_assoc]
        -- This used to be the end of the proof before https://github.com/leanprover/lean4/pull/2644
        /-
          case pos.cons.refl.inr
          G : Type u_1
          inst✝ : Group G
          A B : Subgroup G
          φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
          d : HNNExtension.NormalWord.TransversalPair G A B
          g✝ : G
          w✝ : HNNExtension.NormalWord d
          h1✝ : Membership.mem (d.set { val := Neg.neg ↑(-1), inv := Neg.neg ↑(Inv.inv ( …
          h2✝ : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w✝.toList.head?) …
          hcan : HNNExtension.NormalWord.Cancels (-1) (HNNExtension.NormalWord.cons g✝ { …
          a✝ : Eq (HNNExtension.NormalWord.cons g✝ { val := Neg.neg ↑(-1), inv := Neg.ne …
          ⊢ Eq (HMul.hMul (HNNExtension.of ↑(φ.symm ⟨g✝, ⋯⟩)) (HNNExtension.NormalWord.R …
        -/
        erw [equiv_symm_eq_conj]
        /-
          case pos.cons.refl.inr
          G : Type u_1
          inst✝ : Group G
          A B : Subgroup G
          φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
          d : HNNExtension.NormalWord.TransversalPair G A B
          g✝ : G
          w✝ : HNNExtension.NormalWord d
          h1✝ : Membership.mem (d.set { val := Neg.neg ↑(-1), inv := Neg.neg ↑(Inv.inv ( …
          h2✝ : ∀ (u' : Units Int), Membership.mem (Option.map Prod.fst w✝.toList.head?) …
          hcan : HNNExtension.NormalWord.Cancels (-1) (HNNExtension.NormalWord.cons g✝ { …
          a✝ : Eq (HNNExtension.NormalWord.cons g✝ { val := Neg.neg ↑(-1), inv := Neg.ne …
          ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv HNNExtension.t) (HNNExtension.o …
        -/
        simp [equiv_symm_eq_conj, mul_assoc]
        /-
          🎉 no goals
        -/
    /-
      case neg
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      u : Units Int
      w : HNNExtension.NormalWord d
      hcan : Not (HNNExtension.NormalWord.Cancels u w)
      ⊢ Eq
          (HNNExtension.NormalWord.ReducedWord.prod φ
            (let g' := HNNExtension.NormalWord.unitsSMulGroup φ d u w.head;
              HNNExtension.NormalWord.cons (↑g'.1) u (HSMul.hSMul (HMul.hMul (↑g'.2) …
          (HMul.hMul (HPow.hPow HNNExtension.t ↑u) (HNNExtension.NormalWord.ReducedW …
    -/
  · simp [unitsSMulGroup]
    /-
      case neg
      G : Type u_1
      inst✝ : Group G
      A B : Subgroup G
      φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
      d : HNNExtension.NormalWord.TransversalPair G A B
      u : Units Int
      w : HNNExtension.NormalWord d
      hcan : Not (HNNExtension.NormalWord.Cancels u w)
      ⊢ Eq (HMul.hMul (HNNExtension.of ↑((HNNExtension.toSubgroupEquiv φ u) (⋯.equiv …
    -/
    rcases Int.units_eq_one_or u with (rfl | rfl)
      /-
        case neg.inl
        G : Type u_1
        inst✝ : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        d : HNNExtension.NormalWord.TransversalPair G A B
        w : HNNExtension.NormalWord d
        hcan : Not (HNNExtension.NormalWord.Cancels 1 w)
        ⊢ Eq (HMul.hMul (HNNExtension.of ↑((HNNExtension.toSubgroupEquiv φ 1) (⋯.equiv …
      -/
    · simp [equiv_eq_conj, mul_assoc, (d.compl _).equiv_snd_eq_inv_mul]
      -- This used to be the end of the proof before https://github.com/leanprover/lean4/pull/2644
      /-
        case neg.inl
        G : Type u_1
        inst✝ : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        d : HNNExtension.NormalWord.TransversalPair G A B
        w : HNNExtension.NormalWord d
        hcan : Not (HNNExtension.NormalWord.Cancels 1 w)
        ⊢ Eq (HMul.hMul (HNNExtension.of ↑(⋯.equiv w.head).1) (HMul.hMul (HNNExtension …
      -/
      erw [(d.compl 1).equiv_snd_eq_inv_mul]
      /-
        case neg.inl
        G : Type u_1
        inst✝ : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        d : HNNExtension.NormalWord.TransversalPair G A B
        w : HNNExtension.NormalWord d
        hcan : Not (HNNExtension.NormalWord.Cancels 1 w)
        ⊢ Eq (HMul.hMul (HNNExtension.of ↑(⋯.equiv w.head).1) (HMul.hMul (HNNExtension …
      -/
      simp [equiv_eq_conj, mul_assoc, (d.compl _).equiv_snd_eq_inv_mul]
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        G : Type u_1
        inst✝ : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        d : HNNExtension.NormalWord.TransversalPair G A B
        w : HNNExtension.NormalWord d
        hcan : Not (HNNExtension.NormalWord.Cancels (-1) w)
        ⊢ Eq (HMul.hMul (HNNExtension.of ↑((HNNExtension.toSubgroupEquiv φ (-1)) (⋯.eq …
      -/
    · simp [equiv_symm_eq_conj, mul_assoc, (d.compl _).equiv_snd_eq_inv_mul]
      -- This used to be the end of the proof before https://github.com/leanprover/lean4/pull/2644
      /-
        case neg.inr
        G : Type u_1
        inst✝ : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        d : HNNExtension.NormalWord.TransversalPair G A B
        w : HNNExtension.NormalWord d
        hcan : Not (HNNExtension.NormalWord.Cancels (-1) w)
        ⊢ Eq (HMul.hMul (HNNExtension.of ↑(φ.symm (⋯.equiv w.head).1)) (HMul.hMul (Inv …
      -/
      erw [equiv_symm_eq_conj, (d.compl (-1)).equiv_snd_eq_inv_mul]
      /-
        case neg.inr
        G : Type u_1
        inst✝ : Group G
        A B : Subgroup G
        φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
        d : HNNExtension.NormalWord.TransversalPair G A B
        w : HNNExtension.NormalWord d
        hcan : Not (HNNExtension.NormalWord.Cancels (-1) w)
        ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv HNNExtension.t) (HNNExtension.o …
      -/
      simp [equiv_symm_eq_conj, mul_assoc, (d.compl _).equiv_snd_eq_inv_mul]
      /-
        🎉 no goals
      -/


@[simp]
theorem prod_empty : (empty : NormalWord d).prod φ = 1 := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    d : HNNExtension.NormalWord.TransversalPair G A B
    ⊢ Eq (HNNExtension.NormalWord.ReducedWord.prod φ HNNExtension.NormalWord.empty …
  -/
  simp [ReducedWord.prod]
  /-
    🎉 no goals
  -/


@[simp]
theorem prod_smul (g : HNNExtension G A B φ) (w : NormalWord d) :
    (g • w).prod φ = g * w.prod φ := by
  induction g using induction_on generalizing w with
  | of => simp [of_smul_eq_smul]
  | t => simp [t_smul_eq_unitsSMul, prod_unitsSMul, mul_assoc]
  | mul => simp_all [mul_smul, mul_assoc]
  | inv x ih =>
    rw [← mul_right_inj x, ← ih]
    simp


@[simp]
theorem prod_smul_empty (w : NormalWord d) :
    (w.prod φ) • empty = w := by
  induction w using consRecOn with
  | ofGroup => simp [ofGroup, ReducedWord.prod, of_smul_eq_smul, group_smul_def]
  | cons g u w h1 h2 ih =>
    rw [prod_cons, ← mul_assoc, mul_smul, ih, mul_smul, t_pow_smul_eq_unitsSMul,
      of_smul_eq_smul, unitsSMul]
    rw [dif_neg (not_cancels_of_cons_hyp u w h2)]
    -- The next 3 lines were a single `simp [...]` before https://github.com/leanprover/lean4/pull/2644
    simp only [unitsSMulGroup]
    simp_rw [SetLike.coe_sort_coe]
    erw [(d.compl _).equiv_fst_eq_one_of_mem_of_one_mem (one_mem _) h1]
    ext <;> simp
    -- The next 4 were not needed before https://github.com/leanprover/lean4/pull/2644
    erw [(d.compl _).equiv_snd_eq_inv_mul]
    simp_rw [SetLike.coe_sort_coe]
    erw [(d.compl _).equiv_fst_eq_one_of_mem_of_one_mem (one_mem _) h1]
    simp


/-- The equivalence between elements of the HNN extension and words in normal form. -/
noncomputable def equiv : HNNExtension G A B φ ≃ NormalWord d :=
  { toFun := fun g => g • empty,
    invFun := fun w => w.prod φ,
                            /-
                              G : Type u_1
                              inst✝² : Group G
                              A B : Subgroup G
                              φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
                              H : Type u_2
                              inst✝¹ : Group H
                              M : Type u_3
                              inst✝ : Monoid M
                              d : HNNExtension.NormalWord.TransversalPair G A B
                              g : HNNExtension G A B φ
                              ⊢ Eq ((fun w => HNNExtension.NormalWord.ReducedWord.prod φ w.toReducedWord) (( …
                            -/
    left_inv := fun g => by simp [prod_smul]
                            /-
                              🎉 no goals
                            -/
                             /-
                               G : Type u_1
                               inst✝² : Group G
                               A B : Subgroup G
                               φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
                               H : Type u_2
                               inst✝¹ : Group H
                               M : Type u_3
                               inst✝ : Monoid M
                               d : HNNExtension.NormalWord.TransversalPair G A B
                               w : HNNExtension.NormalWord d
                               ⊢ Eq ((fun g => HSMul.hSMul g HNNExtension.NormalWord.empty) ((fun w => HNNExt …
                             -/
    right_inv := fun w => by simp }
                             /-
                               🎉 no goals
                             -/


theorem prod_injective : Injective
    (fun w => w.prod φ : NormalWord d → HNNExtension G A B φ) :=
  (equiv φ d).symm.injective


instance : FaithfulSMul (HNNExtension G A B φ) (NormalWord d) :=
               /-
                 G : Type u_1
                 inst✝² : Group G
                 A B : Subgroup G
                 φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
                 H : Type u_2
                 inst✝¹ : Group H
                 M : Type u_3
                 inst✝ : Monoid M
                 d : HNNExtension.NormalWord.TransversalPair G A B
                 m₁✝ m₂✝ : HNNExtension G A B φ
                 h : ∀ (a : HNNExtension.NormalWord d), Eq (HSMul.hSMul m₁✝ a) (HSMul.hSMul m₂✝ …
                 ⊢ Eq m₁✝ m₂✝
               -/
  ⟨fun h => by simpa using congr_arg (fun w => w.prod φ) (h empty)⟩
               /-
                 🎉 no goals
               -/


theorem of_injective : Function.Injective (of : G → HNNExtension G A B φ) := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    ⊢ Function.Injective ⇑HNNExtension.of
  -/
  rcases TransversalPair.nonempty G A B with ⟨d⟩
  refine Function.Injective.of_comp
    (f := ((· • ·) : HNNExtension G A B φ → NormalWord d → NormalWord d)) ?_
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    d : HNNExtension.NormalWord.TransversalPair G A B
    ⊢ Function.Injective (Function.comp (fun x1 x2 => HSMul.hSMul x1 x2) ⇑HNNExten …
  -/
  intros _ _ h
  exact eq_of_smul_eq_smul (fun w : NormalWord d =>
    by simp_all [funext_iff, of_smul_eq_smul])


theorem exists_normalWord_prod_eq
    (d : TransversalPair G A B) (w : ReducedWord G A B) :
    ∃ w' : NormalWord d, w'.prod φ = w.prod φ ∧
      w'.toList.map Prod.fst = w.toList.map Prod.fst ∧
      ∀ u ∈ w.toList.head?.map Prod.fst,
      w'.head⁻¹ * w.head ∈ toSubgroup A B (-u) := by
  suffices ∀ w : ReducedWord G A B,
      w.head = 1 → ∃ w' : NormalWord d, w'.prod φ = w.prod φ ∧
      w'.toList.map Prod.fst = w.toList.map Prod.fst ∧
      ∀ u ∈ w.toList.head?.map Prod.fst,
      w'.head ∈ toSubgroup A B (-u) by
    by_cases hw1 : w.head = 1
    · simp only [hw1, inv_mem_iff, mul_one]
      exact this w hw1
    · rcases this ⟨1, w.toList, w.chain⟩ rfl with ⟨w', hw'⟩
      exact ⟨w.head • w', by
        simpa [ReducedWord.prod, mul_assoc] using hw'⟩
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    d : HNNExtension.NormalWord.TransversalPair G A B
    w : HNNExtension.NormalWord.ReducedWord G A B
    ⊢ ∀ (w : HNNExtension.NormalWord.ReducedWord G A B), Eq w.head 1 → Exists fun  …
  -/
  intro w hw1
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    d : HNNExtension.NormalWord.TransversalPair G A B
    w✝ w : HNNExtension.NormalWord.ReducedWord G A B
    hw1 : Eq w.head 1
    ⊢ Exists fun w' => And (Eq (HNNExtension.NormalWord.ReducedWord.prod φ w'.toRe …
  -/
  rcases w with ⟨g, l, chain⟩
  /-
    case mk
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    d : HNNExtension.NormalWord.TransversalPair G A B
    w : HNNExtension.NormalWord.ReducedWord G A B
    g : G
    l : List (Prod (Units Int) G)
    chain : List.Chain' (fun a b => Membership.mem (HNNExtension.toSubgroup A B a. …
    hw1 : Eq { head := g, toList := l, chain := chain }.head 1
    ⊢ Exists fun w' => And (Eq (HNNExtension.NormalWord.ReducedWord.prod φ w'.toRe …
  -/
  dsimp at hw1; subst hw1
  induction l with
  | nil =>
    exact
      ⟨{ head := 1
         toList := []
         mem_set := by simp
         chain := List.chain'_nil }, by simp [prod]⟩
  | cons a l ih =>
    rcases ih (List.chain'_cons'.1 chain).2 with ⟨w', hw'1, hw'2, hw'3⟩
    clear ih
    refine ⟨(t^(a.1 : ℤ) * of a.2 : HNNExtension G A B φ) • w', ?_, ?_⟩
    · rw [prod_smul, hw'1]
      simp [ReducedWord.prod]
    · have : ¬ Cancels a.1 (a.2 • w') := by
        simp only [Cancels, group_smul_head, group_smul_toList, Option.map_eq_some',
          Prod.exists, exists_and_right, exists_eq_right, not_and, not_exists]
        intro hS x hx
        have hx' := congr_arg (Option.map Prod.fst) hx
        rw [← List.head?_map, hw'2, List.head?_map, Option.map_some'] at hx'
        have : w'.head ∈ toSubgroup A B a.fst := by
          simpa using hw'3 _ hx'
        rw [mul_mem_cancel_right this] at hS
        have : a.fst = -a.fst := by
          have hl : l ≠ [] := by rintro rfl; simp_all
          have : a.fst = (l.head hl).fst := (List.chain'_cons'.1 chain).1 (l.head hl)
            (List.head?_eq_head _) hS
          rwa [List.head?_eq_head hl, Option.map_some', ← this, Option.some_inj] at hx'
        simp at this
      erw [List.map_cons, mul_smul, of_smul_eq_smul, NormalWord.group_smul_def,
        t_pow_smul_eq_unitsSMul, unitsSMul, dif_neg this, ← hw'2]
      simp [mul_assoc, unitsSMulGroup, (d.compl _).coe_equiv_snd_eq_one_iff_mem]


/-- Two reduced words representing the same element of the `HNNExtension G A B φ` have the same
length corresponding list, with the same pattern of occurrences of `t^1` and `t^(-1)`,
and also the `head` is in the same left coset of `toSubgroup A B (-u)`, where `u : ℤˣ`
is the exponent of the first occurrence of `t` in the word. -/
theorem map_fst_eq_and_of_prod_eq {w₁ w₂ : ReducedWord G A B}
    (hprod : w₁.prod φ = w₂.prod φ) :
    w₁.toList.map Prod.fst = w₂.toList.map Prod.fst ∧
     ∀ u ∈ w₁.toList.head?.map Prod.fst,
      w₁.head⁻¹ * w₂.head ∈ toSubgroup A B (-u) := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    w₁ w₂ : HNNExtension.NormalWord.ReducedWord G A B
    hprod : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁) (HNNExtension.Norma …
    ⊢ And (Eq (List.map Prod.fst w₁.toList) (List.map Prod.fst w₂.toList)) (∀ (u : …
  -/
  rcases TransversalPair.nonempty G A B with ⟨d⟩
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    w₁ w₂ : HNNExtension.NormalWord.ReducedWord G A B
    hprod : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁) (HNNExtension.Norma …
    d : HNNExtension.NormalWord.TransversalPair G A B
    ⊢ And (Eq (List.map Prod.fst w₁.toList) (List.map Prod.fst w₂.toList)) (∀ (u : …
  -/
  rcases exists_normalWord_prod_eq φ d w₁ with ⟨w₁', hw₁'1, hw₁'2, hw₁'3⟩
  /-
    case intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    w₁ w₂ : HNNExtension.NormalWord.ReducedWord G A B
    hprod : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁) (HNNExtension.Norma …
    d : HNNExtension.NormalWord.TransversalPair G A B
    w₁' : HNNExtension.NormalWord d
    hw₁'1 : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁'.toReducedWord) (HNN …
    hw₁'2 : Eq (List.map Prod.fst w₁'.toList) (List.map Prod.fst w₁.toList)
    hw₁'3 : ∀ (u : Units Int), Membership.mem (Option.map Prod.fst w₁.toList.head? …
    ⊢ And (Eq (List.map Prod.fst w₁.toList) (List.map Prod.fst w₂.toList)) (∀ (u : …
  -/
  rcases exists_normalWord_prod_eq φ d w₂ with ⟨w₂', hw₂'1, hw₂'2, hw₂'3⟩
  have : w₁' = w₂' :=
    NormalWord.prod_injective φ d (by dsimp only; rw [hw₁'1, hw₂'1, hprod])
  /-
    case intro.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    w₁ w₂ : HNNExtension.NormalWord.ReducedWord G A B
    hprod : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁) (HNNExtension.Norma …
    d : HNNExtension.NormalWord.TransversalPair G A B
    w₁' : HNNExtension.NormalWord d
    hw₁'1 : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁'.toReducedWord) (HNN …
    hw₁'2 : Eq (List.map Prod.fst w₁'.toList) (List.map Prod.fst w₁.toList)
    hw₁'3 : ∀ (u : Units Int), Membership.mem (Option.map Prod.fst w₁.toList.head? …
    w₂' : HNNExtension.NormalWord d
    hw₂'1 : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₂'.toReducedWord) (HNN …
    hw₂'2 : Eq (List.map Prod.fst w₂'.toList) (List.map Prod.fst w₂.toList)
    hw₂'3 : ∀ (u : Units Int), Membership.mem (Option.map Prod.fst w₂.toList.head? …
    this : Eq w₁' w₂'
    ⊢ And (Eq (List.map Prod.fst w₁.toList) (List.map Prod.fst w₂.toList)) (∀ (u : …
  -/
  subst this
  /-
    case intro.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    w₁ w₂ : HNNExtension.NormalWord.ReducedWord G A B
    hprod : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁) (HNNExtension.Norma …
    d : HNNExtension.NormalWord.TransversalPair G A B
    w₁' : HNNExtension.NormalWord d
    hw₁'1 : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁'.toReducedWord) (HNN …
    hw₁'2 : Eq (List.map Prod.fst w₁'.toList) (List.map Prod.fst w₁.toList)
    hw₁'3 : ∀ (u : Units Int), Membership.mem (Option.map Prod.fst w₁.toList.head? …
    hw₂'1 : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁'.toReducedWord) (HNN …
    hw₂'2 : Eq (List.map Prod.fst w₁'.toList) (List.map Prod.fst w₂.toList)
    hw₂'3 : ∀ (u : Units Int), Membership.mem (Option.map Prod.fst w₂.toList.head? …
    ⊢ And (Eq (List.map Prod.fst w₁.toList) (List.map Prod.fst w₂.toList)) (∀ (u : …
  -/
  refine ⟨by rw [← hw₁'2, hw₂'2], ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    w₁ w₂ : HNNExtension.NormalWord.ReducedWord G A B
    hprod : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁) (HNNExtension.Norma …
    d : HNNExtension.NormalWord.TransversalPair G A B
    w₁' : HNNExtension.NormalWord d
    hw₁'1 : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁'.toReducedWord) (HNN …
    hw₁'2 : Eq (List.map Prod.fst w₁'.toList) (List.map Prod.fst w₁.toList)
    hw₁'3 : ∀ (u : Units Int), Membership.mem (Option.map Prod.fst w₁.toList.head? …
    hw₂'1 : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁'.toReducedWord) (HNN …
    hw₂'2 : Eq (List.map Prod.fst w₁'.toList) (List.map Prod.fst w₂.toList)
    hw₂'3 : ∀ (u : Units Int), Membership.mem (Option.map Prod.fst w₂.toList.head? …
    ⊢ ∀ (u : Units Int), Membership.mem (Option.map Prod.fst w₁.toList.head?) u →  …
  -/
  simp only [← leftCoset_eq_iff] at *
  /-
    case intro.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    w₁ w₂ : HNNExtension.NormalWord.ReducedWord G A B
    hprod : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁) (HNNExtension.Norma …
    d : HNNExtension.NormalWord.TransversalPair G A B
    w₁' : HNNExtension.NormalWord d
    hw₁'1 : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁'.toReducedWord) (HNN …
    hw₁'2 : Eq (List.map Prod.fst w₁'.toList) (List.map Prod.fst w₁.toList)
    hw₂'1 : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁'.toReducedWord) (HNN …
    hw₂'2 : Eq (List.map Prod.fst w₁'.toList) (List.map Prod.fst w₂.toList)
    hw₁'3 : ∀ (u : Units Int), Membership.mem (Option.map Prod.fst w₁.toList.head? …
    hw₂'3 : ∀ (u : Units Int), Membership.mem (Option.map Prod.fst w₂.toList.head? …
    ⊢ ∀ (u : Units Int), Membership.mem (Option.map Prod.fst w₁.toList.head?) u →  …
  -/
  intro u hu
  /-
    case intro.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    w₁ w₂ : HNNExtension.NormalWord.ReducedWord G A B
    hprod : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁) (HNNExtension.Norma …
    d : HNNExtension.NormalWord.TransversalPair G A B
    w₁' : HNNExtension.NormalWord d
    hw₁'1 : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁'.toReducedWord) (HNN …
    hw₁'2 : Eq (List.map Prod.fst w₁'.toList) (List.map Prod.fst w₁.toList)
    hw₂'1 : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁'.toReducedWord) (HNN …
    hw₂'2 : Eq (List.map Prod.fst w₁'.toList) (List.map Prod.fst w₂.toList)
    hw₁'3 : ∀ (u : Units Int), Membership.mem (Option.map Prod.fst w₁.toList.head? …
    hw₂'3 : ∀ (u : Units Int), Membership.mem (Option.map Prod.fst w₂.toList.head? …
    u : Units Int
    hu : Membership.mem (Option.map Prod.fst w₁.toList.head?) u
    ⊢ Eq (HSMul.hSMul w₁.head ↑(HNNExtension.toSubgroup A B (Neg.neg u))) (HSMul.h …
  -/
  rw [← hw₁'3 _ hu, ← hw₂'3 _]
  /-
    case intro.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    w₁ w₂ : HNNExtension.NormalWord.ReducedWord G A B
    hprod : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁) (HNNExtension.Norma …
    d : HNNExtension.NormalWord.TransversalPair G A B
    w₁' : HNNExtension.NormalWord d
    hw₁'1 : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁'.toReducedWord) (HNN …
    hw₁'2 : Eq (List.map Prod.fst w₁'.toList) (List.map Prod.fst w₁.toList)
    hw₂'1 : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w₁'.toReducedWord) (HNN …
    hw₂'2 : Eq (List.map Prod.fst w₁'.toList) (List.map Prod.fst w₂.toList)
    hw₁'3 : ∀ (u : Units Int), Membership.mem (Option.map Prod.fst w₁.toList.head? …
    hw₂'3 : ∀ (u : Units Int), Membership.mem (Option.map Prod.fst w₂.toList.head? …
    u : Units Int
    hu : Membership.mem (Option.map Prod.fst w₁.toList.head?) u
    ⊢ Membership.mem (Option.map Prod.fst w₂.toList.head?) u
  -/
  rwa [← List.head?_map, ← hw₂'2, hw₁'2, List.head?_map]
  /-
    🎉 no goals
  -/


/-- **Britton's Lemma**. Any reduced word whose product is an element of `G`, has no
occurrences of `t`. -/
theorem toList_eq_nil_of_mem_of_range (w : ReducedWord G A B)
    (hw : w.prod φ ∈ (of.range : Subgroup (HNNExtension G A B φ))) :
    w.toList = [] := by
  /-
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    w : HNNExtension.NormalWord.ReducedWord G A B
    hw : Membership.mem HNNExtension.of.range (HNNExtension.NormalWord.ReducedWord …
    ⊢ Eq w.toList List.nil
  -/
  rcases hw with ⟨g, hg⟩
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    w : HNNExtension.NormalWord.ReducedWord G A B
    g : G
    hg : Eq (HNNExtension.of g) (HNNExtension.NormalWord.ReducedWord.prod φ w)
    ⊢ Eq w.toList List.nil
  -/
  let w' : ReducedWord G A B := { ReducedWord.empty G A B with head := g }
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    w : HNNExtension.NormalWord.ReducedWord G A B
    g : G
    hg : Eq (HNNExtension.of g) (HNNExtension.NormalWord.ReducedWord.prod φ w)
    w' : HNNExtension.NormalWord.ReducedWord G A B :=
      let __src := HNNExtension.NormalWord.ReducedWord.empty G A B;
      { head := g, toList := __src.toList, chain := ⋯ }
    ⊢ Eq w.toList List.nil
  -/
  have : w.prod φ = w'.prod φ := by simp [w', ReducedWord.prod, hg]
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    A B : Subgroup G
    φ : MulEquiv (Subtype fun x => Membership.mem A x) (Subtype fun x => Membershi …
    w : HNNExtension.NormalWord.ReducedWord G A B
    g : G
    hg : Eq (HNNExtension.of g) (HNNExtension.NormalWord.ReducedWord.prod φ w)
    w' : HNNExtension.NormalWord.ReducedWord G A B :=
      let __src := HNNExtension.NormalWord.ReducedWord.empty G A B;
      { head := g, toList := __src.toList, chain := ⋯ }
    this : Eq (HNNExtension.NormalWord.ReducedWord.prod φ w) (HNNExtension.NormalW …
    ⊢ Eq w.toList List.nil
  -/
  simpa [w'] using (map_fst_eq_and_of_prod_eq φ this).1
  /-
    🎉 no goals
  -/



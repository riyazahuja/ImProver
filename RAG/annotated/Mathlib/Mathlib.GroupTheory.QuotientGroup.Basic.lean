open scoped Pointwise in
@[to_additive]
theorem sound (U : Set (G ⧸ N)) (g : N.op) :
    g • (mk' N) ⁻¹' U = (mk' N) ⁻¹' U := by
  /-
    G : Type u
    inst✝ : Group G
    N : Subgroup G
    nN : N.Normal
    U : Set (HasQuotient.Quotient G N)
    g : Subtype fun x => Membership.mem N.op x
    ⊢ Eq (HSMul.hSMul g (Set.preimage (⇑(QuotientGroup.mk' N)) U)) (Set.preimage ( …
  -/
  ext x
  /-
    case h
    G : Type u
    inst✝ : Group G
    N : Subgroup G
    nN : N.Normal
    U : Set (HasQuotient.Quotient G N)
    g : Subtype fun x => Membership.mem N.op x
    x : G
    ⊢ Iff (Membership.mem (HSMul.hSMul g (Set.preimage (⇑(QuotientGroup.mk' N)) U) …
  -/
  simp only [Set.mem_preimage, Set.mem_smul_set_iff_inv_smul_mem]
  /-
    case h
    G : Type u
    inst✝ : Group G
    N : Subgroup G
    nN : N.Normal
    U : Set (HasQuotient.Quotient G N)
    g : Subtype fun x => Membership.mem N.op x
    x : G
    ⊢ Iff (Membership.mem U ((QuotientGroup.mk' N) (HSMul.hSMul (Inv.inv g) x))) ( …
  -/
  congr! 1
  /-
    case h.a.h.e'_5
    G : Type u
    inst✝ : Group G
    N : Subgroup G
    nN : N.Normal
    U : Set (HasQuotient.Quotient G N)
    g : Subtype fun x => Membership.mem N.op x
    x : G
    ⊢ Eq ((QuotientGroup.mk' N) (HSMul.hSMul (Inv.inv g) x)) ((QuotientGroup.mk' N …
  -/
  exact Quotient.sound ⟨g⁻¹, rfl⟩
  /-
    🎉 no goals
  -/

-- for commutative groups we don't need normality assumption


local notation " Q " => G ⧸ N


@[to_additive (attr := simp)]
theorem mk_prod {G ι : Type*} [CommGroup G] (N : Subgroup G) (s : Finset ι) {f : ι → G} :
    ((Finset.prod s f : G) : G ⧸ N) = Finset.prod s (fun i => (f i : G ⧸ N)) :=
  map_prod (QuotientGroup.mk' N) _ _


@[to_additive QuotientAddGroup.strictMono_comap_prod_map]
theorem strictMono_comap_prod_map :
    StrictMono fun H : Subgroup G ↦ (H.comap N.subtype, H.map (mk' N)) :=
  strictMono_comap_prod_image N


/-- The induced map from the quotient by the kernel to the codomain. -/
@[to_additive "The induced map from the quotient by the kernel to the codomain."]
def kerLift : G ⧸ ker φ →* H :=
  lift _ φ fun _g => mem_ker.mp


@[to_additive (attr := simp)]
theorem kerLift_mk (g : G) : (kerLift φ) g = φ g :=
  lift_mk _ _ _


@[to_additive (attr := simp)]
theorem kerLift_mk' (g : G) : (kerLift φ) (mk g) = φ g :=
  lift_mk' _ _ _


@[to_additive]
theorem kerLift_injective : Injective (kerLift φ) := fun a b =>
  Quotient.inductionOn₂' a b fun a b (h : φ a = φ b) =>
                          /-
                            G : Type u
                            inst✝¹ : Group G
                            H : Type v
                            inst✝ : Group H
                            φ : MonoidHom G H
                            a✝ b✝ : HasQuotient.Quotient G φ.ker
                            a b : G
                            h : Eq (φ a) (φ b)
                            ⊢ (QuotientGroup.leftRel φ.ker) a b
                          -/
    Quotient.sound' <| by rw [leftRel_apply, mem_ker, φ.map_mul, ← h, φ.map_inv, inv_mul_cancel]
                          /-
                            🎉 no goals
                          -/

-- Note that `ker φ` isn't definitionally `ker (φ.rangeRestrict)`
-- so there is a bit of annoying code duplication here

/-- The induced map from the quotient by the kernel to the range. -/
@[to_additive "The induced map from the quotient by the kernel to the range."]
def rangeKerLift : G ⧸ ker φ →* φ.range :=
                                                      /-
                                                        G : Type u
                                                        inst✝² : Group G
                                                        N : Subgroup G
                                                        nN : N.Normal
                                                        H : Type v
                                                        inst✝¹ : Group H
                                                        M : Type x
                                                        inst✝ : Monoid M
                                                        φ : MonoidHom G H
                                                        g : G
                                                        hg : Membership.mem φ.ker g
                                                        ⊢ Membership.mem φ.rangeRestrict.ker g
                                                      -/
  lift _ φ.rangeRestrict fun g hg => mem_ker.mp <| by rwa [ker_rangeRestrict]
                                                      /-
                                                        🎉 no goals
                                                      -/


@[to_additive]
theorem rangeKerLift_injective : Injective (rangeKerLift φ) := fun a b =>
  Quotient.inductionOn₂' a b fun a b (h : φ.rangeRestrict a = φ.rangeRestrict b) =>
    Quotient.sound' <| by
      rw [leftRel_apply, ← ker_rangeRestrict, mem_ker, φ.rangeRestrict.map_mul, ← h,
        φ.rangeRestrict.map_inv, inv_mul_cancel]


@[to_additive]
theorem rangeKerLift_surjective : Surjective (rangeKerLift φ) := by
  /-
    G : Type u
    inst✝¹ : Group G
    H : Type v
    inst✝ : Group H
    φ : MonoidHom G H
    ⊢ Function.Surjective ⇑(QuotientGroup.rangeKerLift φ)
  -/
  rintro ⟨_, g, rfl⟩
  /-
    case mk.intro
    G : Type u
    inst✝¹ : Group G
    H : Type v
    inst✝ : Group H
    φ : MonoidHom G H
    g : G
    ⊢ Exists fun a => Eq ((QuotientGroup.rangeKerLift φ) a) ⟨φ g, ⋯⟩
  -/
  use mk g
  /-
    case h
    G : Type u
    inst✝¹ : Group G
    H : Type v
    inst✝ : Group H
    φ : MonoidHom G H
    g : G
    ⊢ Eq ((QuotientGroup.rangeKerLift φ) ↑g) ⟨φ g, ⋯⟩
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- **Noether's first isomorphism theorem** (a definition): the canonical isomorphism between
`G/(ker φ)` to `range φ`. -/
@[to_additive "The first isomorphism theorem (a definition): the canonical isomorphism between
`G/(ker φ)` to `range φ`."]
noncomputable def quotientKerEquivRange : G ⧸ ker φ ≃* range φ :=
  MulEquiv.ofBijective (rangeKerLift φ) ⟨rangeKerLift_injective φ, rangeKerLift_surjective φ⟩


/-- The canonical isomorphism `G/(ker φ) ≃* H` induced by a homomorphism `φ : G →* H`
with a right inverse `ψ : H → G`. -/
@[to_additive (attr := simps) "The canonical isomorphism `G/(ker φ) ≃+ H` induced by a homomorphism
`φ : G →+ H` with a right inverse `ψ : H → G`."]
def quotientKerEquivOfRightInverse (ψ : H → G) (hφ : RightInverse ψ φ) : G ⧸ ker φ ≃* H :=
  { kerLift φ with
    toFun := kerLift φ
    invFun := mk ∘ ψ
                                                 /-
                                                   G : Type u
                                                   inst✝² : Group G
                                                   N : Subgroup G
                                                   nN : N.Normal
                                                   H : Type v
                                                   inst✝¹ : Group H
                                                   M : Type x
                                                   inst✝ : Monoid M
                                                   φ : MonoidHom G H
                                                   ψ : H → G
                                                   hφ : Function.RightInverse ψ ⇑φ
                                                   x : HasQuotient.Quotient G φ.ker
                                                   ⊢ Eq ((QuotientGroup.kerLift φ) (Function.comp QuotientGroup.mk ψ ((QuotientGr …
                                                 -/
    left_inv := fun x => kerLift_injective φ (by rw [Function.comp_apply, kerLift_mk', hφ])
                                                 /-
                                                   🎉 no goals
                                                 -/
    right_inv := hφ }


/-- The canonical isomorphism `G/⊥ ≃* G`. -/
@[to_additive (attr := simps!) "The canonical isomorphism `G/⊥ ≃+ G`."]
def quotientBot : G ⧸ (⊥ : Subgroup G) ≃* G :=
  quotientKerEquivOfRightInverse (MonoidHom.id G) id fun _x => rfl


/-- The canonical isomorphism `G/(ker φ) ≃* H` induced by a surjection `φ : G →* H`.

For a `computable` version, see `QuotientGroup.quotientKerEquivOfRightInverse`.
-/
@[to_additive "The canonical isomorphism `G/(ker φ) ≃+ H` induced by a surjection `φ : G →+ H`.
For a `computable` version, see `QuotientAddGroup.quotientKerEquivOfRightInverse`."]
noncomputable def quotientKerEquivOfSurjective (hφ : Surjective φ) : G ⧸ ker φ ≃* H :=
  quotientKerEquivOfRightInverse φ _ hφ.hasRightInverse.choose_spec


/-- If two normal subgroups `M` and `N` of `G` are the same, their quotient groups are
isomorphic. -/
@[to_additive "If two normal subgroups `M` and `N` of `G` are the same, their quotient groups are
isomorphic."]
def quotientMulEquivOfEq {M N : Subgroup G} [M.Normal] [N.Normal] (h : M = N) : G ⧸ M ≃* G ⧸ N :=
  { Subgroup.quotientEquivOfEq h with
    map_mul' := fun q r => Quotient.inductionOn₂' q r fun _g _h => rfl }


@[to_additive (attr := simp)]
theorem quotientMulEquivOfEq_mk {M N : Subgroup G} [M.Normal] [N.Normal] (h : M = N) (x : G) :
    QuotientGroup.quotientMulEquivOfEq h (QuotientGroup.mk x) = QuotientGroup.mk x :=
  rfl


/-- Let `A', A, B', B` be subgroups of `G`. If `A' ≤ B'` and `A ≤ B`,
then there is a map `A / (A' ⊓ A) →* B / (B' ⊓ B)` induced by the inclusions. -/
@[to_additive "Let `A', A, B', B` be subgroups of `G`. If `A' ≤ B'` and `A ≤ B`, then there is a map
`A / (A' ⊓ A) →+ B / (B' ⊓ B)` induced by the inclusions."]
def quotientMapSubgroupOfOfLe {A' A B' B : Subgroup G} [_hAN : (A'.subgroupOf A).Normal]
    [_hBN : (B'.subgroupOf B).Normal] (h' : A' ≤ B') (h : A ≤ B) :
    A ⧸ A'.subgroupOf A →* B ⧸ B'.subgroupOf B :=
  map _ _ (Subgroup.inclusion h) <| Subgroup.comap_mono h'


@[to_additive (attr := simp)]
theorem quotientMapSubgroupOfOfLe_mk {A' A B' B : Subgroup G} [_hAN : (A'.subgroupOf A).Normal]
    [_hBN : (B'.subgroupOf B).Normal] (h' : A' ≤ B') (h : A ≤ B) (x : A) :
    quotientMapSubgroupOfOfLe h' h x = ↑(Subgroup.inclusion h x : B) :=
  rfl


/-- Let `A', A, B', B` be subgroups of `G`.
If `A' = B'` and `A = B`, then the quotients `A / (A' ⊓ A)` and `B / (B' ⊓ B)` are isomorphic.

Applying this equiv is nicer than rewriting along the equalities, since the type of
`(A'.subgroupOf A : Subgroup A)` depends on `A`.
-/
@[to_additive "Let `A', A, B', B` be subgroups of `G`. If `A' = B'` and `A = B`, then the quotients
`A / (A' ⊓ A)` and `B / (B' ⊓ B)` are isomorphic. Applying this equiv is nicer than rewriting along
the equalities, since the type of `(A'.addSubgroupOf A : AddSubgroup A)` depends on `A`. "]
def equivQuotientSubgroupOfOfEq {A' A B' B : Subgroup G} [hAN : (A'.subgroupOf A).Normal]
    [hBN : (B'.subgroupOf B).Normal] (h' : A' = B') (h : A = B) :
    A ⧸ A'.subgroupOf A ≃* B ⧸ B'.subgroupOf B :=
  (quotientMapSubgroupOfOfLe h'.le h.le).toMulEquiv (quotientMapSubgroupOfOfLe h'.ge h.ge)
        /-
          G : Type u
          inst✝² : Group G
          N : Subgroup G
          nN : N.Normal
          H : Type v
          inst✝¹ : Group H
          M : Type x
          inst✝ : Monoid M
          φ : MonoidHom G H
          A' A B' B : Subgroup G
          hAN : (A'.subgroupOf A).Normal
          hBN : (B'.subgroupOf B).Normal
          h' : Eq A' B'
          h : Eq A B
          ⊢ Eq ((QuotientGroup.quotientMapSubgroupOfOfLe ⋯ ⋯).comp (QuotientGroup.quotie …
        -/
    (by ext ⟨x, hx⟩; rfl)
                     /-
                       🎉 no goals
                     -/
        /-
          G : Type u
          inst✝² : Group G
          N : Subgroup G
          nN : N.Normal
          H : Type v
          inst✝¹ : Group H
          M : Type x
          inst✝ : Monoid M
          φ : MonoidHom G H
          A' A B' B : Subgroup G
          hAN : (A'.subgroupOf A).Normal
          hBN : (B'.subgroupOf B).Normal
          h' : Eq A' B'
          h : Eq A B
          ⊢ Eq ((QuotientGroup.quotientMapSubgroupOfOfLe ⋯ ⋯).comp (QuotientGroup.quotie …
        -/
    (by ext ⟨x, hx⟩; rfl)
                     /-
                       🎉 no goals
                     -/


/-- The map of quotients by powers of an integer induced by a group homomorphism. -/
@[to_additive "The map of quotients by multiples of an integer induced by an additive group
homomorphism."]
def homQuotientZPowOfHom :
    A ⧸ (zpowGroupHom n : A →* A).range →* B ⧸ (zpowGroupHom n : B →* B).range :=
  lift _ ((mk' _).comp f) fun g ⟨h, (hg : h ^ n = g)⟩ =>
    (eq_one_iff _).mpr ⟨f h, by
      /-
        G : Type u
        inst✝⁵ : Group G
        N : Subgroup G
        nN : N.Normal
        H : Type v
        inst✝⁴ : Group H
        M : Type x
        inst✝³ : Monoid M
        φ : MonoidHom G H
        A B C : Type u
        inst✝² : CommGroup A
        inst✝¹ : CommGroup B
        inst✝ : CommGroup C
        f : MonoidHom A B
        g✝ : MonoidHom B A
        e : MulEquiv A B
        d : MulEquiv B C
        n : Int
        g : A
        x✝ : Membership.mem (zpowGroupHom n).range g
        h : A
        hg : Eq (HPow.hPow h n) g
        ⊢ Eq ((zpowGroupHom n) (f h)) (f g)
      -/
      simp only [← hg, map_zpow, zpowGroupHom_apply]⟩
      /-
        🎉 no goals
      -/


@[to_additive (attr := simp)]
theorem homQuotientZPowOfHom_id : homQuotientZPowOfHom (MonoidHom.id A) n = MonoidHom.id _ :=
  monoidHom_ext _ rfl


@[to_additive (attr := simp)]
theorem homQuotientZPowOfHom_comp :
    homQuotientZPowOfHom (f.comp g) n =
      (homQuotientZPowOfHom f n).comp (homQuotientZPowOfHom g n) :=
  monoidHom_ext _ rfl


@[to_additive (attr := simp)]
theorem homQuotientZPowOfHom_comp_of_rightInverse (i : Function.RightInverse g f) :
    (homQuotientZPowOfHom f n).comp (homQuotientZPowOfHom g n) = MonoidHom.id _ :=
  monoidHom_ext _ <| MonoidHom.ext fun x => congrArg _ <| i x


/-- The equivalence of quotients by powers of an integer induced by a group isomorphism. -/
@[to_additive "The equivalence of quotients by multiples of an integer induced by an additive group
isomorphism."]
def equivQuotientZPowOfEquiv :
    A ⧸ (zpowGroupHom n : A →* A).range ≃* B ⧸ (zpowGroupHom n : B →* B).range :=
  MonoidHom.toMulEquiv _ _
    (homQuotientZPowOfHom_comp_of_rightInverse (e.symm : B →* A) (e : A →* B) n e.left_inv)
    (homQuotientZPowOfHom_comp_of_rightInverse (e : A →* B) (e.symm : B →* A) n e.right_inv)
    -- Porting note: had to explicitly coerce the `MulEquiv`s to `MonoidHom`s


@[to_additive (attr := simp)]
theorem equivQuotientZPowOfEquiv_refl :
    MulEquiv.refl (A ⧸ (zpowGroupHom n : A →* A).range) =
      equivQuotientZPowOfEquiv (MulEquiv.refl A) n := by
  /-
    A : Type u
    inst✝ : CommGroup A
    n : Int
    ⊢ Eq (MulEquiv.refl (HasQuotient.Quotient A (zpowGroupHom n).range)) (Quotient …
  -/
  ext x
  /-
    case h
    A : Type u
    inst✝ : CommGroup A
    n : Int
    x : HasQuotient.Quotient A (zpowGroupHom n).range
    ⊢ Eq ((MulEquiv.refl (HasQuotient.Quotient A (zpowGroupHom n).range)) x) ((Quo …
  -/
  rw [← Quotient.out_eq' x]
  /-
    case h
    A : Type u
    inst✝ : CommGroup A
    n : Int
    x : HasQuotient.Quotient A (zpowGroupHom n).range
    ⊢ Eq ((MulEquiv.refl (HasQuotient.Quotient A (zpowGroupHom n).range)) (Quotien …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem equivQuotientZPowOfEquiv_symm :
    (equivQuotientZPowOfEquiv e n).symm = equivQuotientZPowOfEquiv e.symm n :=
  rfl


@[to_additive (attr := simp)]
theorem equivQuotientZPowOfEquiv_trans :
    (equivQuotientZPowOfEquiv e n).trans (equivQuotientZPowOfEquiv d n) =
      equivQuotientZPowOfEquiv (e.trans d) n := by
  /-
    A B C : Type u
    inst✝² : CommGroup A
    inst✝¹ : CommGroup B
    inst✝ : CommGroup C
    e : MulEquiv A B
    d : MulEquiv B C
    n : Int
    ⊢ Eq ((QuotientGroup.equivQuotientZPowOfEquiv e n).trans (QuotientGroup.equivQ …
  -/
  ext x
  /-
    case h
    A B C : Type u
    inst✝² : CommGroup A
    inst✝¹ : CommGroup B
    inst✝ : CommGroup C
    e : MulEquiv A B
    d : MulEquiv B C
    n : Int
    x : HasQuotient.Quotient A (zpowGroupHom n).range
    ⊢ Eq (((QuotientGroup.equivQuotientZPowOfEquiv e n).trans (QuotientGroup.equiv …
  -/
  rw [← Quotient.out_eq' x]
  /-
    case h
    A B C : Type u
    inst✝² : CommGroup A
    inst✝¹ : CommGroup B
    inst✝ : CommGroup C
    e : MulEquiv A B
    d : MulEquiv B C
    n : Int
    x : HasQuotient.Quotient A (zpowGroupHom n).range
    ⊢ Eq (((QuotientGroup.equivQuotientZPowOfEquiv e n).trans (QuotientGroup.equiv …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- **Noether's second isomorphism theorem**: given two subgroups `H` and `N` of a group `G`, where
`N` is normal, defines an isomorphism between `H/(H ∩ N)` and `(HN)/N`. -/
@[to_additive "The second isomorphism theorem: given two subgroups `H` and `N` of a group `G`, where
`N` is normal, defines an isomorphism between `H/(H ∩ N)` and `(H + N)/N`"]
noncomputable def quotientInfEquivProdNormalQuotient (H N : Subgroup G) [N.Normal] :
    H ⧸ N.subgroupOf H ≃* _ ⧸ N.subgroupOf (H ⊔ N) :=
  let
    φ :-- φ is the natural homomorphism H →* (HN)/N.
      H →*
      _ ⧸ N.subgroupOf (H ⊔ N) :=
    (mk' <| N.subgroupOf (H ⊔ N)).comp (inclusion le_sup_left)
  have φ_surjective : Surjective φ := fun x =>
    x.inductionOn' <| by
      /-
        G : Type u
        inst✝³ : Group G
        N✝ : Subgroup G
        nN : N✝.Normal
        H✝ : Type v
        inst✝² : Group H✝
        M : Type x
        inst✝¹ : Monoid M
        φ✝ : MonoidHom G H✝
        H N : Subgroup G
        inst✝ : N.Normal
        φ : MonoidHom (Subtype fun x => Membership.mem H x) (HasQuotient.Quotient (Sub …
        x : HasQuotient.Quotient (Subtype fun x => Membership.mem (Max.max H N) x) (N. …
        ⊢ ∀ (a : Subtype fun x => Membership.mem (Max.max H N) x), Exists fun a_1 => E …
      -/
      rintro ⟨y, hy : y ∈ (H ⊔ N)⟩
      /-
        case mk
        G : Type u
        inst✝³ : Group G
        N✝ : Subgroup G
        nN : N✝.Normal
        H✝ : Type v
        inst✝² : Group H✝
        M : Type x
        inst✝¹ : Monoid M
        φ✝ : MonoidHom G H✝
        H N : Subgroup G
        inst✝ : N.Normal
        φ : MonoidHom (Subtype fun x => Membership.mem H x) (HasQuotient.Quotient (Sub …
        x : HasQuotient.Quotient (Subtype fun x => Membership.mem (Max.max H N) x) (N. …
        y : G
        hy : Membership.mem (Max.max H N) y
        ⊢ Exists fun a => Eq (φ a) (Quotient.mk'' ⟨y, hy⟩)
      -/
      rw [← SetLike.mem_coe] at hy
      /-
        case mk
        G : Type u
        inst✝³ : Group G
        N✝ : Subgroup G
        nN : N✝.Normal
        H✝ : Type v
        inst✝² : Group H✝
        M : Type x
        inst✝¹ : Monoid M
        φ✝ : MonoidHom G H✝
        H N : Subgroup G
        inst✝ : N.Normal
        φ : MonoidHom (Subtype fun x => Membership.mem H x) (HasQuotient.Quotient (Sub …
        x : HasQuotient.Quotient (Subtype fun x => Membership.mem (Max.max H N) x) (N. …
        y : G
        hy✝ : Membership.mem (Max.max H N) y
        hy : Membership.mem (↑(Max.max H N)) y
        ⊢ Exists fun a => Eq (φ a) (Quotient.mk'' ⟨y, hy✝⟩)
      -/
      rw [mul_normal H N] at hy
      /-
        case mk
        G : Type u
        inst✝³ : Group G
        N✝ : Subgroup G
        nN : N✝.Normal
        H✝ : Type v
        inst✝² : Group H✝
        M : Type x
        inst✝¹ : Monoid M
        φ✝ : MonoidHom G H✝
        H N : Subgroup G
        inst✝ : N.Normal
        φ : MonoidHom (Subtype fun x => Membership.mem H x) (HasQuotient.Quotient (Sub …
        x : HasQuotient.Quotient (Subtype fun x => Membership.mem (Max.max H N) x) (N. …
        y : G
        hy✝ : Membership.mem (Max.max H N) y
        hy : Membership.mem (HMul.hMul ↑H ↑N) y
        ⊢ Exists fun a => Eq (φ a) (Quotient.mk'' ⟨y, hy✝⟩)
      -/
      rcases hy with ⟨h, hh, n, hn, rfl⟩
      /-
        case mk.intro.intro.intro.intro
        G : Type u
        inst✝³ : Group G
        N✝ : Subgroup G
        nN : N✝.Normal
        H✝ : Type v
        inst✝² : Group H✝
        M : Type x
        inst✝¹ : Monoid M
        φ✝ : MonoidHom G H✝
        H N : Subgroup G
        inst✝ : N.Normal
        φ : MonoidHom (Subtype fun x => Membership.mem H x) (HasQuotient.Quotient (Sub …
        x : HasQuotient.Quotient (Subtype fun x => Membership.mem (Max.max H N) x) (N. …
        h : G
        hh : Membership.mem (↑H) h
        n : G
        hn : Membership.mem (↑N) n
        hy : Membership.mem (Max.max H N) ((fun x1 x2 => HMul.hMul x1 x2) h n)
        ⊢ Exists fun a => Eq (φ a) (Quotient.mk'' ⟨(fun x1 x2 => HMul.hMul x1 x2) h n, …
      -/
      use ⟨h, hh⟩
      let _ : Setoid ↑(H ⊔ N) :=
        (@leftRel ↑(H ⊔ N) (H ⊔ N : Subgroup G).toGroup (N.subgroupOf (H ⊔ N)))
      -- Porting note: Lean couldn't find this automatically
      /-
        case h
        G : Type u
        inst✝³ : Group G
        N✝ : Subgroup G
        nN : N✝.Normal
        H✝ : Type v
        inst✝² : Group H✝
        M : Type x
        inst✝¹ : Monoid M
        φ✝ : MonoidHom G H✝
        H N : Subgroup G
        inst✝ : N.Normal
        φ : MonoidHom (Subtype fun x => Membership.mem H x) (HasQuotient.Quotient (Sub …
        x : HasQuotient.Quotient (Subtype fun x => Membership.mem (Max.max H N) x) (N. …
        h : G
        hh : Membership.mem (↑H) h
        n : G
        hn : Membership.mem (↑N) n
        hy : Membership.mem (Max.max H N) ((fun x1 x2 => HMul.hMul x1 x2) h n)
        x✝ : Setoid (Subtype fun x => Membership.mem (Max.max H N) x) := QuotientGroup …
        ⊢ Eq (φ ⟨h, hh⟩) (Quotient.mk'' ⟨(fun x1 x2 => HMul.hMul x1 x2) h n, hy⟩)
      -/
      refine Quotient.eq.mpr ?_
      /-
        case h
        G : Type u
        inst✝³ : Group G
        N✝ : Subgroup G
        nN : N✝.Normal
        H✝ : Type v
        inst✝² : Group H✝
        M : Type x
        inst✝¹ : Monoid M
        φ✝ : MonoidHom G H✝
        H N : Subgroup G
        inst✝ : N.Normal
        φ : MonoidHom (Subtype fun x => Membership.mem H x) (HasQuotient.Quotient (Sub …
        x : HasQuotient.Quotient (Subtype fun x => Membership.mem (Max.max H N) x) (N. …
        h : G
        hh : Membership.mem (↑H) h
        n : G
        hn : Membership.mem (↑N) n
        hy : Membership.mem (Max.max H N) ((fun x1 x2 => HMul.hMul x1 x2) h n)
        x✝ : Setoid (Subtype fun x => Membership.mem (Max.max H N) x) := QuotientGroup …
        ⊢ (QuotientGroup.leftRel (N.subgroupOf (Max.max H N))) ((Subgroup.inclusion ⋯) …
      -/
      change leftRel _ _ _
      /-
        case h
        G : Type u
        inst✝³ : Group G
        N✝ : Subgroup G
        nN : N✝.Normal
        H✝ : Type v
        inst✝² : Group H✝
        M : Type x
        inst✝¹ : Monoid M
        φ✝ : MonoidHom G H✝
        H N : Subgroup G
        inst✝ : N.Normal
        φ : MonoidHom (Subtype fun x => Membership.mem H x) (HasQuotient.Quotient (Sub …
        x : HasQuotient.Quotient (Subtype fun x => Membership.mem (Max.max H N) x) (N. …
        h : G
        hh : Membership.mem (↑H) h
        n : G
        hn : Membership.mem (↑N) n
        hy : Membership.mem (Max.max H N) ((fun x1 x2 => HMul.hMul x1 x2) h n)
        x✝ : Setoid (Subtype fun x => Membership.mem (Max.max H N) x) := QuotientGroup …
        ⊢ (QuotientGroup.leftRel (N.subgroupOf (Max.max H N))) ((Subgroup.inclusion ⋯) …
      -/
      rw [leftRel_apply]
      /-
        case h
        G : Type u
        inst✝³ : Group G
        N✝ : Subgroup G
        nN : N✝.Normal
        H✝ : Type v
        inst✝² : Group H✝
        M : Type x
        inst✝¹ : Monoid M
        φ✝ : MonoidHom G H✝
        H N : Subgroup G
        inst✝ : N.Normal
        φ : MonoidHom (Subtype fun x => Membership.mem H x) (HasQuotient.Quotient (Sub …
        x : HasQuotient.Quotient (Subtype fun x => Membership.mem (Max.max H N) x) (N. …
        h : G
        hh : Membership.mem (↑H) h
        n : G
        hn : Membership.mem (↑N) n
        hy : Membership.mem (Max.max H N) ((fun x1 x2 => HMul.hMul x1 x2) h n)
        x✝ : Setoid (Subtype fun x => Membership.mem (Max.max H N) x) := QuotientGroup …
        ⊢ Membership.mem (N.subgroupOf (Max.max H N)) (HMul.hMul (Inv.inv ((Subgroup.i …
      -/
      change h⁻¹ * (h * n) ∈ N
      /-
        case h
        G : Type u
        inst✝³ : Group G
        N✝ : Subgroup G
        nN : N✝.Normal
        H✝ : Type v
        inst✝² : Group H✝
        M : Type x
        inst✝¹ : Monoid M
        φ✝ : MonoidHom G H✝
        H N : Subgroup G
        inst✝ : N.Normal
        φ : MonoidHom (Subtype fun x => Membership.mem H x) (HasQuotient.Quotient (Sub …
        x : HasQuotient.Quotient (Subtype fun x => Membership.mem (Max.max H N) x) (N. …
        h : G
        hh : Membership.mem (↑H) h
        n : G
        hn : Membership.mem (↑N) n
        hy : Membership.mem (Max.max H N) ((fun x1 x2 => HMul.hMul x1 x2) h n)
        x✝ : Setoid (Subtype fun x => Membership.mem (Max.max H N) x) := QuotientGroup …
        ⊢ Membership.mem N (HMul.hMul (Inv.inv h) (HMul.hMul h n))
      -/
      rwa [← mul_assoc, inv_mul_cancel, one_mul]
      /-
        🎉 no goals
      -/
                            /-
                              G : Type u
                              inst✝³ : Group G
                              N✝ : Subgroup G
                              nN : N✝.Normal
                              H✝ : Type v
                              inst✝² : Group H✝
                              M : Type x
                              inst✝¹ : Monoid M
                              φ✝ : MonoidHom G H✝
                              H N : Subgroup G
                              inst✝ : N.Normal
                              φ : MonoidHom (Subtype fun x => Membership.mem H x) (HasQuotient.Quotient (Sub …
                              φ_surjective : Function.Surjective ⇑φ
                              ⊢ Eq (N.subgroupOf H) φ.ker
                            -/
  (quotientMulEquivOfEq (by simp [φ, ← comap_ker])).trans
                            /-
                              🎉 no goals
                            -/
    (quotientKerEquivOfSurjective φ φ_surjective)


@[to_additive]
instance map_normal : (M.map (QuotientGroup.mk' N)).Normal :=
  nM.map _ mk_surjective


/-- The map from the third isomorphism theorem for groups: `(G / N) / (M / N) → G / M`. -/
@[to_additive "The map from the third isomorphism theorem for additive groups:
`(A / N) / (M / N) → A / M`."]
def quotientQuotientEquivQuotientAux : (G ⧸ N) ⧸ M.map (mk' N) →* G ⧸ M :=
  lift (M.map (mk' N)) (map N M (MonoidHom.id G) h)
    (by
      /-
        G : Type u
        inst✝² : Group G
        N : Subgroup G
        nN : N.Normal
        H : Type v
        inst✝¹ : Group H
        M✝ : Type x
        inst✝ : Monoid M✝
        φ : MonoidHom G H
        M : Subgroup G
        nM : M.Normal
        h : LE.le N M
        ⊢ LE.le (Subgroup.map (QuotientGroup.mk' N) M) (QuotientGroup.map N M (MonoidH …
      -/
      rintro _ ⟨x, hx, rfl⟩
      /-
        case intro.intro
        G : Type u
        inst✝² : Group G
        N : Subgroup G
        nN : N.Normal
        H : Type v
        inst✝¹ : Group H
        M✝ : Type x
        inst✝ : Monoid M✝
        φ : MonoidHom G H
        M : Subgroup G
        nM : M.Normal
        h : LE.le N M
        x : G
        hx : Membership.mem (↑M) x
        ⊢ Membership.mem (QuotientGroup.map N M (MonoidHom.id G) h).ker ((QuotientGrou …
      -/
      rw [mem_ker, map_mk' N M _ _ x]
      /-
        case intro.intro
        G : Type u
        inst✝² : Group G
        N : Subgroup G
        nN : N.Normal
        H : Type v
        inst✝¹ : Group H
        M✝ : Type x
        inst✝ : Monoid M✝
        φ : MonoidHom G H
        M : Subgroup G
        nM : M.Normal
        h : LE.le N M
        x : G
        hx : Membership.mem (↑M) x
        ⊢ Eq (↑((MonoidHom.id G) x)) 1
      -/
      exact (QuotientGroup.eq_one_iff _).mpr hx)
      /-
        🎉 no goals
      -/


@[to_additive (attr := simp)]
theorem quotientQuotientEquivQuotientAux_mk (x : G ⧸ N) :
    quotientQuotientEquivQuotientAux N M h x = QuotientGroup.map N M (MonoidHom.id G) h x :=
  QuotientGroup.lift_mk' _ _ x


@[to_additive]
theorem quotientQuotientEquivQuotientAux_mk_mk (x : G) :
    quotientQuotientEquivQuotientAux N M h (x : G ⧸ N) = x :=
  QuotientGroup.lift_mk' (M.map (mk' N)) _ x


/-- **Noether's third isomorphism theorem** for groups: `(G / N) / (M / N) ≃* G / M`. -/
@[to_additive
      "**Noether's third isomorphism theorem** for additive groups: `(A / N) / (M / N) ≃+ A / M`."]
def quotientQuotientEquivQuotient : (G ⧸ N) ⧸ M.map (QuotientGroup.mk' N) ≃* G ⧸ M :=
  MonoidHom.toMulEquiv (quotientQuotientEquivQuotientAux N M h)
    (QuotientGroup.map _ _ (QuotientGroup.mk' N) (Subgroup.le_comap_map _ _))
        /-
          G : Type u
          inst✝² : Group G
          N : Subgroup G
          nN : N.Normal
          H : Type v
          inst✝¹ : Group H
          M✝ : Type x
          inst✝ : Monoid M✝
          φ : MonoidHom G H
          M : Subgroup G
          nM : M.Normal
          h : LE.le N M
          ⊢ Eq ((QuotientGroup.map M (Subgroup.map (QuotientGroup.mk' N) M) (QuotientGro …
        -/
    (by ext; simp)
             /-
               🎉 no goals
             -/
        /-
          G : Type u
          inst✝² : Group G
          N : Subgroup G
          nN : N.Normal
          H : Type v
          inst✝¹ : Group H
          M✝ : Type x
          inst✝ : Monoid M✝
          φ : MonoidHom G H
          M : Subgroup G
          nM : M.Normal
          h : LE.le N M
          ⊢ Eq ((QuotientGroup.quotientQuotientEquivQuotientAux N M h).comp (QuotientGro …
        -/
    (by ext; simp)
             /-
               🎉 no goals
             -/


@[to_additive]
theorem subsingleton_quotient_top : Subsingleton (G ⧸ (⊤ : Subgroup G)) := by
  /-
    G : Type u
    inst✝ : Group G
    ⊢ Subsingleton (HasQuotient.Quotient G Top.top)
  -/
  dsimp [HasQuotient.Quotient, QuotientGroup.instHasQuotientSubgroup, Quotient]
  /-
    G : Type u
    inst✝ : Group G
    ⊢ Subsingleton (Quot ⇑(QuotientGroup.leftRel Top.top))
  -/
  rw [leftRel_eq]
  /-
    G : Type u
    inst✝ : Group G
    ⊢ Subsingleton (Quot fun x y => Membership.mem Top.top (HMul.hMul (Inv.inv x)  …
  -/
  exact Trunc.instSubsingletonTrunc
  /-
    🎉 no goals
  -/


/-- If the quotient by a subgroup gives a singleton then the subgroup is the whole group. -/
@[to_additive "If the quotient by an additive subgroup gives a singleton then the additive subgroup
is the whole additive group."]
theorem subgroup_eq_top_of_subsingleton (H : Subgroup G) (h : Subsingleton (G ⧸ H)) : H = ⊤ :=
  top_unique fun x _ => by
    /-
      G : Type u
      inst✝ : Group G
      H : Subgroup G
      h : Subsingleton (HasQuotient.Quotient G H)
      x : G
      x✝ : Membership.mem Top.top x
      ⊢ Membership.mem H x
    -/
    have this : 1⁻¹ * x ∈ H := QuotientGroup.eq.1 (Subsingleton.elim _ _)
    /-
      G : Type u
      inst✝ : Group G
      H : Subgroup G
      h : Subsingleton (HasQuotient.Quotient G H)
      x : G
      x✝ : Membership.mem Top.top x
      this : Membership.mem H (HMul.hMul (Inv.inv 1) x)
      ⊢ Membership.mem H x
    -/
    rwa [inv_one, one_mul] at this
    /-
      🎉 no goals
    -/


@[to_additive]
theorem comap_comap_center {H₁ : Subgroup G} [H₁.Normal] {H₂ : Subgroup (G ⧸ H₁)} [H₂.Normal] :
    ((Subgroup.center ((G ⧸ H₁) ⧸ H₂)).comap (mk' H₂)).comap (mk' H₁) =
      (Subgroup.center (G ⧸ H₂.comap (mk' H₁))).comap (mk' (H₂.comap (mk' H₁))) := by
  /-
    G : Type u
    inst✝² : Group G
    H₁ : Subgroup G
    inst✝¹ : H₁.Normal
    H₂ : Subgroup (HasQuotient.Quotient G H₁)
    inst✝ : H₂.Normal
    ⊢ Eq (Subgroup.comap (QuotientGroup.mk' H₁) (Subgroup.comap (QuotientGroup.mk' …
  -/
  ext x
  simp only [mk'_apply, Subgroup.mem_comap, Subgroup.mem_center_iff, forall_mk, ← mk_mul,
    eq_iff_div_mem, mk_div]


@[simp]
theorem mk_nat_mul (n : ℕ) (a : R) : ((n * a : R) : R ⧸ N) = n • ↑a := by
  /-
    R : Type u_1
    inst✝¹ : NonAssocRing R
    N : AddSubgroup R
    inst✝ : N.Normal
    n : Nat
    a : R
    ⊢ Eq (↑(HMul.hMul (↑n) a)) (HSMul.hSMul n ↑a)
  -/
  rw [← nsmul_eq_mul, mk_nsmul N a n]
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_int_mul (n : ℤ) (a : R) : ((n * a : R) : R ⧸ N) = n • ↑a := by
  /-
    R : Type u_1
    inst✝¹ : NonAssocRing R
    N : AddSubgroup R
    inst✝ : N.Normal
    n : Int
    a : R
    ⊢ Eq (↑(HMul.hMul (↑n) a)) (HSMul.hSMul n ↑a)
  -/
  rw [← zsmul_eq_mul, mk_zsmul N a n]
  /-
    🎉 no goals
  -/



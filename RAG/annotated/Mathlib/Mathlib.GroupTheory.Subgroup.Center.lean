/-- The center of a group `G` is the set of elements that commute with everything in `G` -/
@[to_additive
      "The center of an additive group `G` is the set of elements that commute with
      everything in `G`"]
def center : Subgroup G :=
  { Submonoid.center G with
    carrier := Set.center G
    inv_mem' := Set.inv_mem_center }


@[to_additive]
theorem coe_center : ↑(center G) = Set.center G :=
  rfl


@[to_additive (attr := simp)]
theorem center_toSubmonoid : (center G).toSubmonoid = Submonoid.center G :=
  rfl


instance center.isCommutative : (center G).IsCommutative :=
  ⟨⟨fun a b => Subtype.ext (b.2.comm a).symm⟩⟩


/-- For a group with zero, the center of the units is the same as the units of the center. -/
@[simps! apply_val_coe symm_apply_coe_val]
def centerUnitsEquivUnitsCenter (G₀ : Type*) [GroupWithZero G₀] :
    Subgroup.center (G₀ˣ) ≃* (Submonoid.center G₀)ˣ where
  toFun := MonoidHom.toHomUnits <|
    { toFun := fun u ↦ ⟨(u : G₀ˣ),
      (Submonoid.mem_center_iff.mpr (fun r ↦ by
          /-
            G : Type u_1
            inst✝¹ : Group G
            G₀ : Type u_2
            inst✝ : GroupWithZero G₀
            u : Subtype fun x => Membership.mem (Subgroup.center (Units G₀)) x
            r : G₀
            ⊢ Eq (HMul.hMul r ↑↑u) (HMul.hMul (↑↑u) r)
          -/
          rcases eq_or_ne r 0 with (rfl | hr)
            /-
              case inl
              G : Type u_1
              inst✝¹ : Group G
              G₀ : Type u_2
              inst✝ : GroupWithZero G₀
              u : Subtype fun x => Membership.mem (Subgroup.center (Units G₀)) x
              ⊢ Eq (HMul.hMul 0 ↑↑u) (HMul.hMul (↑↑u) 0)
            -/
          · rw [mul_zero, zero_mul]
            /-
              🎉 no goals
            -/
          /-
            case inr
            G : Type u_1
            inst✝¹ : Group G
            G₀ : Type u_2
            inst✝ : GroupWithZero G₀
            u : Subtype fun x => Membership.mem (Subgroup.center (Units G₀)) x
            r : G₀
            hr : Ne r 0
            ⊢ Eq (HMul.hMul r ↑↑u) (HMul.hMul (↑↑u) r)
          -/
          exact congrArg Units.val <| (u.2.comm <| Units.mk0 r hr).symm))⟩
          /-
            🎉 no goals
          -/
      map_one' := rfl
      map_mul' := fun _ _ ↦ rfl }
  invFun u := unitsCenterToCenterUnits G₀ u
                   /-
                     G : Type u_1
                     inst✝¹ : Group G
                     G₀ : Type u_2
                     inst✝ : GroupWithZero G₀
                     x✝ : Subtype fun x => Membership.mem (Subgroup.center (Units G₀)) x
                     ⊢ Eq ((fun u => (unitsCenterToCenterUnits G₀) u) ({ toFun := fun u => ⟨↑↑u, ⋯⟩ …
                   -/
  left_inv _ := by ext; rfl
                        /-
                          🎉 no goals
                        -/
                    /-
                      G : Type u_1
                      inst✝¹ : Group G
                      G₀ : Type u_2
                      inst✝ : GroupWithZero G₀
                      x✝ : Units (Subtype fun x => Membership.mem (Submonoid.center G₀) x)
                      ⊢ Eq ({ toFun := fun u => ⟨↑↑u, ⋯⟩, map_one' := ?m.1809, map_mul' := ⋯ }.toHom …
                    -/
  right_inv _ := by ext; rfl
                         /-
                           🎉 no goals
                         -/
  map_mul' := map_mul _


@[to_additive]
theorem mem_center_iff {z : G} : z ∈ center G ↔ ∀ g, g * z = z * g := by
  /-
    G : Type u_1
    inst✝ : Group G
    z : G
    ⊢ Iff (Membership.mem (Subgroup.center G) z) (∀ (g : G), Eq (HMul.hMul g z) (H …
  -/
  rw [← Semigroup.mem_center_iff]
  /-
    G : Type u_1
    inst✝ : Group G
    z : G
    ⊢ Iff (Membership.mem (Subgroup.center G) z) (Membership.mem (Set.center G) z)
  -/
  exact Iff.rfl
  /-
    🎉 no goals
  -/


instance decidableMemCenter (z : G) [Decidable (∀ g, g * z = z * g)] : Decidable (z ∈ center G) :=
  decidable_of_iff' _ mem_center_iff


@[to_additive]
instance centerCharacteristic : (center G).Characteristic := by
  /-
    G : Type u_1
    inst✝ : Group G
    ⊢ (Subgroup.center G).Characteristic
  -/
  refine characteristic_iff_comap_le.mpr fun ϕ g hg => ?_
  /-
    G : Type u_1
    inst✝ : Group G
    ϕ : MulEquiv G G
    g : G
    hg : Membership.mem (Subgroup.comap ϕ.toMonoidHom (Subgroup.center G)) g
    ⊢ Membership.mem (Subgroup.center G) g
  -/
  rw [mem_center_iff]
  /-
    G : Type u_1
    inst✝ : Group G
    ϕ : MulEquiv G G
    g : G
    hg : Membership.mem (Subgroup.comap ϕ.toMonoidHom (Subgroup.center G)) g
    ⊢ ∀ (g_1 : G), Eq (HMul.hMul g_1 g) (HMul.hMul g g_1)
  -/
  intro h
  /-
    G : Type u_1
    inst✝ : Group G
    ϕ : MulEquiv G G
    g : G
    hg : Membership.mem (Subgroup.comap ϕ.toMonoidHom (Subgroup.center G)) g
    h : G
    ⊢ Eq (HMul.hMul h g) (HMul.hMul g h)
  -/
  rw [← ϕ.injective.eq_iff, map_mul, map_mul]
  /-
    G : Type u_1
    inst✝ : Group G
    ϕ : MulEquiv G G
    g : G
    hg : Membership.mem (Subgroup.comap ϕ.toMonoidHom (Subgroup.center G)) g
    h : G
    ⊢ Eq (HMul.hMul (ϕ h) (ϕ g)) (HMul.hMul (ϕ g) (ϕ h))
  -/
  exact (hg.comm (ϕ h)).symm
  /-
    🎉 no goals
  -/


theorem _root_.CommGroup.center_eq_top {G : Type*} [CommGroup G] : center G = ⊤ := by
  /-
    G : Type u_2
    inst✝ : CommGroup G
    ⊢ Eq (Subgroup.center G) Top.top
  -/
  rw [eq_top_iff']
  /-
    G : Type u_2
    inst✝ : CommGroup G
    ⊢ ∀ (x : G), Membership.mem (Subgroup.center G) x
  -/
  intro x
  /-
    G : Type u_2
    inst✝ : CommGroup G
    x : G
    ⊢ Membership.mem (Subgroup.center G) x
  -/
  rw [Subgroup.mem_center_iff]
  /-
    G : Type u_2
    inst✝ : CommGroup G
    x : G
    ⊢ ∀ (g : G), Eq (HMul.hMul g x) (HMul.hMul x g)
  -/
  intro y
  /-
    G : Type u_2
    inst✝ : CommGroup G
    x y : G
    ⊢ Eq (HMul.hMul y x) (HMul.hMul x y)
  -/
  exact mul_comm y x
  /-
    🎉 no goals
  -/


/-- A group is commutative if the center is the whole group -/
def _root_.Group.commGroupOfCenterEqTop (h : center G = ⊤) : CommGroup G :=
  { ‹Group G› with
    mul_comm := by
      /-
        G : Type u_1
        inst✝ : Group G
        h : Eq (Subgroup.center G) Top.top
        ⊢ ∀ (a b : G), Eq (HMul.hMul a b) (HMul.hMul b a)
      -/
      rw [eq_top_iff'] at h
      /-
        G : Type u_1
        inst✝ : Group G
        h : ∀ (x : G), Membership.mem (Subgroup.center G) x
        __src✝ : Group G := inst✝
        ⊢ ∀ (a b : G), Eq (HMul.hMul a b) (HMul.hMul b a)
      -/
      intro x y
      /-
        G : Type u_1
        inst✝ : Group G
        h : ∀ (x : G), Membership.mem (Subgroup.center G) x
        __src✝ : Group G := inst✝
        x y : G
        ⊢ Eq (HMul.hMul x y) (HMul.hMul y x)
      -/
      apply Subgroup.mem_center_iff.mp _ x
      /-
        G : Type u_1
        inst✝ : Group G
        h : ∀ (x : G), Membership.mem (Subgroup.center G) x
        __src✝ : Group G := inst✝
        x y : G
        ⊢ Membership.mem (Subgroup.center G) y
      -/
      exact h y
      /-
        🎉 no goals
      -/
  }


@[to_additive]
theorem center_le_normalizer : center G ≤ H.normalizer := fun x hx y => by
  /-
    G : Type u_1
    inst✝ : Group G
    H : Subgroup G
    x : G
    hx : Membership.mem (Subgroup.center G) x
    y : G
    ⊢ Iff (Membership.mem H y) (Membership.mem H (HMul.hMul (HMul.hMul x y) (Inv.i …
  -/
  simp [← mem_center_iff.mp hx y, mul_assoc]
  /-
    🎉 no goals
  -/


theorem eq_of_left_mem_center {g h : M} (H : IsConj g h) (Hg : g ∈ Set.center M) : g = h := by
  /-
    M : Type u_2
    inst✝ : Monoid M
    g h : M
    H : IsConj g h
    Hg : Membership.mem (Set.center M) g
    ⊢ Eq g h
  -/
  rcases H with ⟨u, hu⟩; rwa [← u.mul_left_inj, Hg.comm u]
                         /-
                           🎉 no goals
                         -/


theorem eq_of_right_mem_center {g h : M} (H : IsConj g h) (Hh : h ∈ Set.center M) : g = h :=
  (H.symm.eq_of_left_mem_center Hh).symm


theorem mk_bijOn (G : Type*) [Group G] :
    Set.BijOn ConjClasses.mk (↑(Subgroup.center G)) (noncenter G)ᶜ := by
  /-
    G : Type u_2
    inst✝ : Group G
    ⊢ Set.BijOn ConjClasses.mk (↑(Subgroup.center G)) (HasCompl.compl (ConjClasses …
  -/
  refine ⟨fun g hg ↦ ?_, fun x hx y _ H ↦ ?_, ?_⟩
    /-
      case refine_1
      G : Type u_2
      inst✝ : Group G
      g : G
      hg : Membership.mem (↑(Subgroup.center G)) g
      ⊢ Membership.mem (HasCompl.compl (ConjClasses.noncenter G)) (ConjClasses.mk g)
    -/
  · simp only [mem_noncenter, Set.compl_def, Set.mem_setOf, Set.not_nontrivial_iff]
    /-
      case refine_1
      G : Type u_2
      inst✝ : Group G
      g : G
      hg : Membership.mem (↑(Subgroup.center G)) g
      ⊢ (ConjClasses.mk g).carrier.Subsingleton
    -/
    intro x hx y hy
    /-
      case refine_1
      G : Type u_2
      inst✝ : Group G
      g : G
      hg : Membership.mem (↑(Subgroup.center G)) g
      x : G
      hx : Membership.mem (ConjClasses.mk g).carrier x
      y : G
      hy : Membership.mem (ConjClasses.mk g).carrier y
      ⊢ Eq x y
    -/
    simp only [mem_carrier_iff_mk_eq, mk_eq_mk_iff_isConj] at hx hy
    /-
      case refine_1
      G : Type u_2
      inst✝ : Group G
      g : G
      hg : Membership.mem (↑(Subgroup.center G)) g
      x y : G
      hx : IsConj x g
      hy : IsConj y g
      ⊢ Eq x y
    -/
    rw [hx.eq_of_right_mem_center hg, hy.eq_of_right_mem_center hg]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_2
      inst✝ : Group G
      x : G
      hx : Membership.mem (↑(Subgroup.center G)) x
      y : G
      x✝ : Membership.mem (↑(Subgroup.center G)) y
      H : Eq (ConjClasses.mk x) (ConjClasses.mk y)
      ⊢ Eq x y
    -/
  · rw [mk_eq_mk_iff_isConj] at H
    /-
      case refine_2
      G : Type u_2
      inst✝ : Group G
      x : G
      hx : Membership.mem (↑(Subgroup.center G)) x
      y : G
      x✝ : Membership.mem (↑(Subgroup.center G)) y
      H : IsConj x y
      ⊢ Eq x y
    -/
    exact H.eq_of_left_mem_center hx
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      G : Type u_2
      inst✝ : Group G
      ⊢ Set.SurjOn ConjClasses.mk (↑(Subgroup.center G)) (HasCompl.compl (ConjClasse …
    -/
  · rintro ⟨g⟩ hg
    /-
      case refine_3.mk
      G : Type u_2
      inst✝ : Group G
      a✝ : ConjClasses G
      g : G
      hg : Membership.mem (HasCompl.compl (ConjClasses.noncenter G)) (Quot.mk (⇑(IsC …
      ⊢ Membership.mem (Set.image ConjClasses.mk ↑(Subgroup.center G)) (Quot.mk (⇑(I …
    -/
    refine ⟨g, ?_, rfl⟩
    /-
      case refine_3.mk
      G : Type u_2
      inst✝ : Group G
      a✝ : ConjClasses G
      g : G
      hg : Membership.mem (HasCompl.compl (ConjClasses.noncenter G)) (Quot.mk (⇑(IsC …
      ⊢ Membership.mem (↑(Subgroup.center G)) g
    -/
    simp only [mem_noncenter, Set.compl_def, Set.mem_setOf, Set.not_nontrivial_iff] at hg
    /-
      case refine_3.mk
      G : Type u_2
      inst✝ : Group G
      a✝ : ConjClasses G
      g : G
      hg : (ConjClasses.carrier (Quot.mk (⇑(IsConj.setoid G)) g)).Subsingleton
      ⊢ Membership.mem (↑(Subgroup.center G)) g
    -/
    rw [SetLike.mem_coe, Subgroup.mem_center_iff]
    /-
      case refine_3.mk
      G : Type u_2
      inst✝ : Group G
      a✝ : ConjClasses G
      g : G
      hg : (ConjClasses.carrier (Quot.mk (⇑(IsConj.setoid G)) g)).Subsingleton
      ⊢ ∀ (g_1 : G), Eq (HMul.hMul g_1 g) (HMul.hMul g g_1)
    -/
    intro h
    /-
      case refine_3.mk
      G : Type u_2
      inst✝ : Group G
      a✝ : ConjClasses G
      g : G
      hg : (ConjClasses.carrier (Quot.mk (⇑(IsConj.setoid G)) g)).Subsingleton
      h : G
      ⊢ Eq (HMul.hMul h g) (HMul.hMul g h)
    -/
    rw [← mul_inv_eq_iff_eq_mul]
    /-
      case refine_3.mk
      G : Type u_2
      inst✝ : Group G
      a✝ : ConjClasses G
      g : G
      hg : (ConjClasses.carrier (Quot.mk (⇑(IsConj.setoid G)) g)).Subsingleton
      h : G
      ⊢ Eq (HMul.hMul (HMul.hMul h g) (Inv.inv h)) g
    -/
    refine hg ?_ mem_carrier_mk
    /-
      case refine_3.mk
      G : Type u_2
      inst✝ : Group G
      a✝ : ConjClasses G
      g : G
      hg : (ConjClasses.carrier (Quot.mk (⇑(IsConj.setoid G)) g)).Subsingleton
      h : G
      ⊢ Membership.mem (ConjClasses.carrier (Quot.mk (⇑(IsConj.setoid G)) g)) (HMul. …
    -/
    rw [mem_carrier_iff_mk_eq]
    /-
      case refine_3.mk
      G : Type u_2
      inst✝ : Group G
      a✝ : ConjClasses G
      g : G
      hg : (ConjClasses.carrier (Quot.mk (⇑(IsConj.setoid G)) g)).Subsingleton
      h : G
      ⊢ Eq (ConjClasses.mk (HMul.hMul (HMul.hMul h g) (Inv.inv h))) (Quot.mk (⇑(IsCo …
    -/
    apply mk_eq_mk_iff_isConj.mpr
    /-
      case refine_3.mk
      G : Type u_2
      inst✝ : Group G
      a✝ : ConjClasses G
      g : G
      hg : (ConjClasses.carrier (Quot.mk (⇑(IsConj.setoid G)) g)).Subsingleton
      h : G
      ⊢ IsConj (HMul.hMul (HMul.hMul h g) (Inv.inv h)) g
    -/
    rw [isConj_comm, isConj_iff]
    /-
      case refine_3.mk
      G : Type u_2
      inst✝ : Group G
      a✝ : ConjClasses G
      g : G
      hg : (ConjClasses.carrier (Quot.mk (⇑(IsConj.setoid G)) g)).Subsingleton
      h : G
      ⊢ Exists fun c => Eq (HMul.hMul (HMul.hMul c g) (Inv.inv c)) (HMul.hMul (HMul. …
    -/
    exact ⟨h, rfl⟩
    /-
      🎉 no goals
    -/



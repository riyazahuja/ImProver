@[to_additive (attr := simp)]
theorem mabs_mem_iff {S G} [Group G] [LinearOrder G] {_ : SetLike S G}
    [InvMemClass S G] {H : S} {x : G} : |x|ₘ ∈ H ↔ x ∈ H := by
  /-
    S : Type u_1
    G : Type u_2
    inst✝² : Group G
    inst✝¹ : LinearOrder G
    x✝ : SetLike S G
    inst✝ : InvMemClass S G
    H : S
    x : G
    ⊢ Iff (Membership.mem H (mabs x)) (Membership.mem H x)
  -/
                          /-
                            🎉 no goals
                          -/
  cases mabs_choice x <;> simp [*]
                          /-
                            🎉 no goals
                          -/


@[to_additive]
instance : IsModularLattice (Subgroup C) :=
  ⟨fun {x} y z xz a ha => by
    /-
      C : Type u_1
      inst✝ : CommGroup C
      x y z : Subgroup C
      xz : LE.le x z
      a : C
      ha : Membership.mem (Min.min (Max.max x y) z) a
      ⊢ Membership.mem (Max.max x (Min.min y z)) a
    -/
    rw [mem_inf, mem_sup] at ha
    /-
      C : Type u_1
      inst✝ : CommGroup C
      x y z : Subgroup C
      xz : LE.le x z
      a : C
      ha : And (Exists fun y_1 => And (Membership.mem x y_1) (Exists fun z => And (M …
      ⊢ Membership.mem (Max.max x (Min.min y z)) a
    -/
    rcases ha with ⟨⟨b, hb, c, hc, rfl⟩, haz⟩
    /-
      case intro.intro.intro.intro.intro
      C : Type u_1
      inst✝ : CommGroup C
      x y z : Subgroup C
      xz : LE.le x z
      b : C
      hb : Membership.mem x b
      c : C
      hc : Membership.mem y c
      haz : Membership.mem z (HMul.hMul b c)
      ⊢ Membership.mem (Max.max x (Min.min y z)) (HMul.hMul b c)
    -/
    rw [mem_sup]
    /-
      case intro.intro.intro.intro.intro
      C : Type u_1
      inst✝ : CommGroup C
      x y z : Subgroup C
      xz : LE.le x z
      b : C
      hb : Membership.mem x b
      c : C
      hc : Membership.mem y c
      haz : Membership.mem z (HMul.hMul b c)
      ⊢ Exists fun y_1 => And (Membership.mem x y_1) (Exists fun z_1 => And (Members …
    -/
    exact ⟨b, hb, c, mem_inf.2 ⟨hc, (mul_mem_cancel_left (xz hb)).1 haz⟩, rfl⟩⟩
    /-
      🎉 no goals
    -/


/-- In a group that satisfies the normalizer condition, every maximal subgroup is normal -/
theorem NormalizerCondition.normal_of_coatom (hnc : NormalizerCondition G) (hmax : IsCoatom H) :
    H.Normal :=
  normalizer_eq_top_iff.mp (hmax.2 _ (hnc H (lt_top_iff_ne_top.mpr hmax.1)))


@[simp]
theorem isCoatom_comap {H : Type*} [Group H] (f : G ≃* H) {K : Subgroup H} :
    IsCoatom (Subgroup.comap (f : G →* H) K) ↔ IsCoatom K :=
  OrderIso.isCoatom_iff (f.comapSubgroup) K


@[simp]
theorem isCoatom_map (f : G ≃* H) {K : Subgroup G} :
    IsCoatom (Subgroup.map (f : G →* H) K) ↔ IsCoatom K :=
  OrderIso.isCoatom_iff (f.mapSubgroup) K


lemma isCoatom_comap_of_surjective
    {H : Type*} [Group H] {φ : G →* H} (hφ : Function.Surjective φ)
    {M : Subgroup H} (hM : IsCoatom M) : IsCoatom (M.comap φ) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    H : Type u_2
    inst✝ : Group H
    φ : MonoidHom G H
    hφ : Function.Surjective ⇑φ
    M : Subgroup H
    hM : IsCoatom M
    ⊢ IsCoatom (Subgroup.comap φ M)
  -/
  refine And.imp (fun hM ↦ ?_) (fun hM ↦ ?_) hM
    /-
      case refine_1
      G : Type u_1
      inst✝¹ : Group G
      H : Type u_2
      inst✝ : Group H
      φ : MonoidHom G H
      hφ : Function.Surjective ⇑φ
      M : Subgroup H
      hM✝ : IsCoatom M
      hM : Ne M Top.top
      ⊢ Ne (Subgroup.comap φ M) Top.top
    -/
  · rwa [← (comap_injective hφ).ne_iff, comap_top] at hM
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u_1
      inst✝¹ : Group G
      H : Type u_2
      inst✝ : Group H
      φ : MonoidHom G H
      hφ : Function.Surjective ⇑φ
      M : Subgroup H
      hM✝ : IsCoatom M
      hM : ∀ (b : Subgroup H), LT.lt M b → Eq b Top.top
      ⊢ ∀ (b : Subgroup G), LT.lt (Subgroup.comap φ M) b → Eq b Top.top
    -/
  · intro K hK
    /-
      case refine_2
      G : Type u_1
      inst✝¹ : Group G
      H : Type u_2
      inst✝ : Group H
      φ : MonoidHom G H
      hφ : Function.Surjective ⇑φ
      M : Subgroup H
      hM✝ : IsCoatom M
      hM : ∀ (b : Subgroup H), LT.lt M b → Eq b Top.top
      K : Subgroup G
      hK : LT.lt (Subgroup.comap φ M) K
      ⊢ Eq K Top.top
    -/
    specialize hM (K.map φ)
    /-
      case refine_2
      G : Type u_1
      inst✝¹ : Group G
      H : Type u_2
      inst✝ : Group H
      φ : MonoidHom G H
      hφ : Function.Surjective ⇑φ
      M : Subgroup H
      hM✝ : IsCoatom M
      K : Subgroup G
      hK : LT.lt (Subgroup.comap φ M) K
      hM : LT.lt M (Subgroup.map φ K) → Eq (Subgroup.map φ K) Top.top
      ⊢ Eq K Top.top
    -/
    rw [← comap_lt_comap_of_surjective hφ, ← (comap_injective hφ).eq_iff] at hM
    /-
      case refine_2
      G : Type u_1
      inst✝¹ : Group G
      H : Type u_2
      inst✝ : Group H
      φ : MonoidHom G H
      hφ : Function.Surjective ⇑φ
      M : Subgroup H
      hM✝ : IsCoatom M
      K : Subgroup G
      hK : LT.lt (Subgroup.comap φ M) K
      hM : LT.lt (Subgroup.comap φ M) (Subgroup.comap φ (Subgroup.map φ K)) → Eq (Su …
      ⊢ Eq K Top.top
    -/
    rw [comap_map_eq_self ((M.ker_le_comap φ).trans hK.le), comap_top] at hM
    /-
      case refine_2
      G : Type u_1
      inst✝¹ : Group G
      H : Type u_2
      inst✝ : Group H
      φ : MonoidHom G H
      hφ : Function.Surjective ⇑φ
      M : Subgroup H
      hM✝ : IsCoatom M
      K : Subgroup G
      hK : LT.lt (Subgroup.comap φ M) K
      hM : LT.lt (Subgroup.comap φ M) K → Eq K Top.top
      ⊢ Eq K Top.top
    -/
    exact hM hK
    /-
      🎉 no goals
    -/


/-- A subgroup of an `OrderedCommGroup` is an `OrderedCommGroup`. -/
@[to_additive "An additive subgroup of an `AddOrderedCommGroup` is an `AddOrderedCommGroup`."]
instance (priority := 75) toOrderedCommGroup [OrderedCommGroup G]
    [SubgroupClass S G] (H : S) : OrderedCommGroup H :=
  Subtype.coe_injective.orderedCommGroup _ rfl (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) fun _ _ => rfl

-- Prefer subclasses of `Group` over subclasses of `SubgroupClass`.

/-- A subgroup of a `LinearOrderedCommGroup` is a `LinearOrderedCommGroup`. -/
@[to_additive
      "An additive subgroup of a `LinearOrderedAddCommGroup` is a
        `LinearOrderedAddCommGroup`."]
instance (priority := 75) toLinearOrderedCommGroup [LinearOrderedCommGroup G]
    [SubgroupClass S G] (H : S) : LinearOrderedCommGroup H :=
  Subtype.coe_injective.linearOrderedCommGroup _ rfl (fun _ _ => rfl) (fun _ => rfl)
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) fun _ _ => rfl


/-- A subgroup of an `OrderedCommGroup` is an `OrderedCommGroup`. -/
@[to_additive "An `AddSubgroup` of an `AddOrderedCommGroup` is an `AddOrderedCommGroup`."]
instance toOrderedCommGroup [OrderedCommGroup G] (H : Subgroup G) :
    OrderedCommGroup H :=
  Subtype.coe_injective.orderedCommGroup _ rfl (fun _ _ => rfl) (fun _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) fun _ _ => rfl


/-- A subgroup of a `LinearOrderedCommGroup` is a `LinearOrderedCommGroup`. -/
@[to_additive
      "An `AddSubgroup` of a `LinearOrderedAddCommGroup` is a
        `LinearOrderedAddCommGroup`."]
instance toLinearOrderedCommGroup [LinearOrderedCommGroup G] (H : Subgroup G) :
    LinearOrderedCommGroup H :=
  Subtype.coe_injective.linearOrderedCommGroup _ rfl (fun _ _ => rfl) (fun _ => rfl)
    (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) fun _ _ => rfl


@[to_additive]
lemma Subsemigroup.strictMono_topEquiv {G : Type*} [OrderedCommMonoid G] :
    StrictMono (topEquiv (M := G)) := fun _ _ ↦ id


@[to_additive]
lemma MulEquiv.strictMono_subsemigroupCongr {G : Type*} [OrderedCommMonoid G] {S T : Subsemigroup G}
    (h : S = T) : StrictMono (subsemigroupCongr h) := fun _ _ ↦ id


@[to_additive]
lemma MulEquiv.strictMono_symm {G G' : Type*} [LinearOrderedCommMonoid G]
    [LinearOrderedCommMonoid G'] {e : G ≃* G'} (he : StrictMono e) : StrictMono e.symm := by
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : LinearOrderedCommMonoid G
    inst✝ : LinearOrderedCommMonoid G'
    e : MulEquiv G G'
    he : StrictMono ⇑e
    ⊢ StrictMono ⇑e.symm
  -/
  intro
  /-
    G : Type u_1
    G' : Type u_2
    inst✝¹ : LinearOrderedCommMonoid G
    inst✝ : LinearOrderedCommMonoid G'
    e : MulEquiv G G'
    he : StrictMono ⇑e
    a✝ : G'
    ⊢ ∀ ⦃b : G'⦄, LT.lt a✝ b → LT.lt (e.symm a✝) (e.symm b)
  -/
  simp [← he.lt_iff_lt]
  /-
    🎉 no goals
  -/


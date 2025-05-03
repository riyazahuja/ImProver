/-- The `centralizer` of `s` is the subgroup of `g : G` commuting with every `h : s`. -/
@[to_additive
      "The `centralizer` of `s` is the additive subgroup of `g : G` commuting with every `h : s`."]
def centralizer (s : Set G) : Subgroup G :=
  { Submonoid.centralizer s with
    carrier := Set.centralizer s
    inv_mem' := Set.inv_mem_centralizer }


@[to_additive]
theorem mem_centralizer_iff {g : G} {s : Set G} : g ∈ centralizer s ↔ ∀ h ∈ s, h * g = g * h :=
  Iff.rfl


@[to_additive]
theorem mem_centralizer_iff_commutator_eq_one {g : G} {s : Set G} :
    g ∈ centralizer s ↔ ∀ h ∈ s, h * g * h⁻¹ * g⁻¹ = 1 := by
  /-
    G : Type u_1
    inst✝ : Group G
    g : G
    s : Set G
    ⊢ Iff (Membership.mem (Subgroup.centralizer s) g) (∀ (h : G), Membership.mem s …
  -/
  simp only [mem_centralizer_iff, mul_inv_eq_iff_eq_mul, one_mul]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma mem_centralizer_singleton_iff {g k : G} :
    k ∈ Subgroup.centralizer {g} ↔ k * g = g * k := by
  /-
    G : Type u_1
    inst✝ : Group G
    g k : G
    ⊢ Iff (Membership.mem (Subgroup.centralizer (Singleton.singleton g)) k) (Eq (H …
  -/
  simp only [mem_centralizer_iff, Set.mem_singleton_iff, forall_eq]
  /-
    G : Type u_1
    inst✝ : Group G
    g k : G
    ⊢ Iff (Eq (HMul.hMul g k) (HMul.hMul k g)) (Eq (HMul.hMul k g) (HMul.hMul g k))
  -/
  exact eq_comm
  /-
    🎉 no goals
  -/


@[to_additive]
theorem centralizer_univ : centralizer Set.univ = center G :=
  SetLike.ext' (Set.centralizer_univ G)


@[to_additive]
theorem le_centralizer_iff : H ≤ centralizer K ↔ K ≤ centralizer H :=
  ⟨fun h x hx _y hy => (h hy x hx).symm, fun h x hx _y hy => (h hy x hx).symm⟩


@[to_additive]
theorem center_le_centralizer (s) : center G ≤ centralizer s :=
  Set.center_subset_centralizer s


@[to_additive]
theorem centralizer_le {s t : Set G} (h : s ⊆ t) : centralizer t ≤ centralizer s :=
  Submonoid.centralizer_le h


@[to_additive (attr := simp)]
theorem centralizer_eq_top_iff_subset {s : Set G} : centralizer s = ⊤ ↔ s ⊆ center G :=
  SetLike.ext'_iff.trans Set.centralizer_eq_top_iff_subset


@[to_additive]
instance Centralizer.characteristic [hH : H.Characteristic] :
    (centralizer (H : Set G)).Characteristic := by
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    hH : H.Characteristic
    ⊢ (Subgroup.centralizer ↑H).Characteristic
  -/
  refine Subgroup.characteristic_iff_comap_le.mpr fun ϕ g hg h hh => ϕ.injective ?_
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    hH : H.Characteristic
    ϕ : MulEquiv G G
    g : G
    hg : Membership.mem (Subgroup.comap ϕ.toMonoidHom (Subgroup.centralizer ↑H)) g
    h : G
    hh : Membership.mem (↑H) h
    ⊢ Eq (ϕ (HMul.hMul h g)) (ϕ (HMul.hMul g h))
  -/
  rw [map_mul, map_mul]
  /-
    G : Type u_1
    inst✝ : Group G
    H K : Subgroup G
    hH : H.Characteristic
    ϕ : MulEquiv G G
    g : G
    hg : Membership.mem (Subgroup.comap ϕ.toMonoidHom (Subgroup.centralizer ↑H)) g
    h : G
    hh : Membership.mem (↑H) h
    ⊢ Eq (HMul.hMul (ϕ h) (ϕ g)) (HMul.hMul (ϕ g) (ϕ h))
  -/
  exact hg (ϕ h) (Subgroup.characteristic_iff_le_comap.mp hH ϕ hh)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem le_centralizer_iff_isCommutative : K ≤ centralizer K ↔ K.IsCommutative :=
  ⟨fun h => ⟨⟨fun x y => Subtype.ext (h y.2 x x.2)⟩⟩,
    fun h x hx y hy => congr_arg Subtype.val (h.1.1 ⟨y, hy⟩ ⟨x, hx⟩)⟩


@[to_additive]
theorem le_centralizer [h : H.IsCommutative] : H ≤ centralizer H :=
  le_centralizer_iff_isCommutative.mpr h


variable {H} in
@[to_additive]
lemma closure_le_centralizer_centralizer (s : Set G) :
    closure s ≤ centralizer (centralizer s) :=
  closure_le _ |>.mpr Set.subset_centralizer_centralizer


/-- If all the elements of a set `s` commute, then `closure s` is a commutative group. -/
@[to_additive
      "If all the elements of a set `s` commute, then `closure s` is an additive
      commutative group."]
abbrev closureCommGroupOfComm {k : Set G} (hcomm : ∀ x ∈ k, ∀ y ∈ k, x * y = y * x) :
    CommGroup (closure k) :=
  { (closure k).toGroup with
    mul_comm := fun ⟨_, h₁⟩ ⟨_, h₂⟩ ↦
      have := closure_le_centralizer_centralizer k
      Subtype.ext <| Set.centralizer_centralizer_comm_of_comm hcomm _ (this h₁) _ (this h₂) }


/-- The conjugation action of N(H) on H. -/
@[simps]
instance : MulDistribMulAction H.normalizer H where
  smul g h := ⟨g * h * g⁻¹, (g.2 h).mp h.2⟩
                   /-
                     G : Type u_1
                     inst✝ : Group G
                     H K : Subgroup G
                     g : Subtype fun x => Membership.mem H x
                     ⊢ Eq (HSMul.hSMul 1 g) g
                   -/
  one_smul g := by simp [HSMul.hSMul]
                   /-
                     🎉 no goals
                   -/
                 /-
                   G : Type u_1
                   inst✝ : Group G
                   H K : Subgroup G
                   ⊢ ∀ (x y : Subtype fun x => Membership.mem H.normalizer x) (b : Subtype fun x  …
                 -/
  mul_smul := by simp [HSMul.hSMul, mul_assoc]
                 /-
                   🎉 no goals
                 -/
                 /-
                   G : Type u_1
                   inst✝ : Group G
                   H K : Subgroup G
                   ⊢ ∀ (r : Subtype fun x => Membership.mem H.normalizer x), Eq (HSMul.hSMul r 1) 1
                 -/
                 /-
                   G : Type u_1
                   inst✝ : Group G
                   H K : Subgroup G
                   ⊢ ∀ (r : Subtype fun x => Membership.mem H.normalizer x) (x y : Subtype fun x  …
                 -/
  smul_one := by simp [HSMul.hSMul]
                 /-
                   🎉 no goals
                 -/
                 /-
                   🎉 no goals
                 -/
  smul_mul := by simp [HSMul.hSMul]


/-- The homomorphism N(H) → Aut(H) with kernel C(H). -/
@[simps!]
def normalizerMonoidHom : H.normalizer →* MulAut H :=
  MulDistribMulAction.toMulAut H.normalizer H


theorem normalizerMonoidHom_ker :
    H.normalizerMonoidHom.ker = (Subgroup.centralizer H).subgroupOf H.normalizer := by
  simp [Subgroup.ext_iff, DFunLike.ext_iff, Subtype.ext_iff,
    mem_subgroupOf, mem_centralizer_iff, eq_mul_inv_iff_mul_eq, eq_comm]



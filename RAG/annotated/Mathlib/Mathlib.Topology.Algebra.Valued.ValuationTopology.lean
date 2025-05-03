/-- The basis of open subgroups for the topology on a ring determined by a valuation. -/
theorem subgroups_basis : RingSubgroupsBasis fun γ : Γ₀ˣ => (v.ltAddSubgroup γ : AddSubgroup R) :=
  { inter := by
      /-
        R : Type u
        inst✝¹ : Ring R
        Γ₀ : Type v
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation R Γ₀
        ⊢ ∀ (i j : Units Γ₀), Exists fun k => LE.le (v.ltAddSubgroup k) (Min.min (v.lt …
      -/
      rintro γ₀ γ₁
      /-
        R : Type u
        inst✝¹ : Ring R
        Γ₀ : Type v
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation R Γ₀
        γ₀ γ₁ : Units Γ₀
        ⊢ Exists fun k => LE.le (v.ltAddSubgroup k) (Min.min (v.ltAddSubgroup γ₀) (v.l …
      -/
      use min γ₀ γ₁
      simp only [ltAddSubgroup, Units.min_val, Units.val_le_val, lt_min_iff,
        AddSubgroup.mk_le_mk, setOf_subset_setOf, le_inf_iff, and_imp, imp_self, implies_true,
        forall_const, and_true]
      /-
        case h
        R : Type u
        inst✝¹ : Ring R
        Γ₀ : Type v
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation R Γ₀
        γ₀ γ₁ : Units Γ₀
        ⊢ ∀ (a : R), LT.lt (v a) ↑γ₀ → LT.lt (v a) ↑γ₁ → LT.lt (v a) ↑γ₀
      -/
      tauto
      /-
        🎉 no goals
      -/
    mul := by
      /-
        R : Type u
        inst✝¹ : Ring R
        Γ₀ : Type v
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation R Γ₀
        ⊢ ∀ (i : Units Γ₀), Exists fun j => HasSubset.Subset (HMul.hMul ↑(v.ltAddSubgr …
      -/
      rintro γ
      /-
        R : Type u
        inst✝¹ : Ring R
        Γ₀ : Type v
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation R Γ₀
        γ : Units Γ₀
        ⊢ Exists fun j => HasSubset.Subset (HMul.hMul ↑(v.ltAddSubgroup j) ↑(v.ltAddSu …
      -/
      cases' exists_square_le γ with γ₀ h
      /-
        case intro
        R : Type u
        inst✝¹ : Ring R
        Γ₀ : Type v
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation R Γ₀
        γ γ₀ : Units Γ₀
        h : LE.le (HMul.hMul γ₀ γ₀) γ
        ⊢ Exists fun j => HasSubset.Subset (HMul.hMul ↑(v.ltAddSubgroup j) ↑(v.ltAddSu …
      -/
      use γ₀
      /-
        case h
        R : Type u
        inst✝¹ : Ring R
        Γ₀ : Type v
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation R Γ₀
        γ γ₀ : Units Γ₀
        h : LE.le (HMul.hMul γ₀ γ₀) γ
        ⊢ HasSubset.Subset (HMul.hMul ↑(v.ltAddSubgroup γ₀) ↑(v.ltAddSubgroup γ₀)) ↑(v …
      -/
      rintro - ⟨r, r_in, s, s_in, rfl⟩
      /-
        case h.intro.intro.intro.intro
        R : Type u
        inst✝¹ : Ring R
        Γ₀ : Type v
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation R Γ₀
        γ γ₀ : Units Γ₀
        h : LE.le (HMul.hMul γ₀ γ₀) γ
        r : R
        r_in : Membership.mem (↑(v.ltAddSubgroup γ₀)) r
        s : R
        s_in : Membership.mem (↑(v.ltAddSubgroup γ₀)) s
        ⊢ Membership.mem (↑(v.ltAddSubgroup γ)) ((fun x1 x2 => HMul.hMul x1 x2) r s)
      -/
      simp only [ltAddSubgroup, AddSubgroup.coe_set_mk, mem_setOf_eq] at r_in s_in
      calc
        (v (r * s) : Γ₀) = v r * v s := Valuation.map_mul _ _ _
        _ < γ₀ * γ₀ := by gcongr <;> exact zero_le'
        _ ≤ γ := mod_cast h
    leftMul := by
      /-
        R : Type u
        inst✝¹ : Ring R
        Γ₀ : Type v
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation R Γ₀
        ⊢ ∀ (x : R) (i : Units Γ₀), Exists fun j => HasSubset.Subset (↑(v.ltAddSubgrou …
      -/
      rintro x γ
      /-
        R : Type u
        inst✝¹ : Ring R
        Γ₀ : Type v
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation R Γ₀
        x : R
        γ : Units Γ₀
        ⊢ Exists fun j => HasSubset.Subset (↑(v.ltAddSubgroup j)) (Set.preimage (fun x …
      -/
      rcases GroupWithZero.eq_zero_or_unit (v x) with (Hx | ⟨γx, Hx⟩)
        /-
          case inl
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ : Units Γ₀
          Hx : Eq (v x) 0
          ⊢ Exists fun j => HasSubset.Subset (↑(v.ltAddSubgroup j)) (Set.preimage (fun x …
        -/
      · use (1 : Γ₀ˣ)
        /-
          case h
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ : Units Γ₀
          Hx : Eq (v x) 0
          ⊢ HasSubset.Subset (↑(v.ltAddSubgroup 1)) (Set.preimage (fun x_1 => HMul.hMul  …
        -/
        rintro y _
        /-
          case h
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ : Units Γ₀
          Hx : Eq (v x) 0
          y : R
          a✝ : Membership.mem (↑(v.ltAddSubgroup 1)) y
          ⊢ Membership.mem (Set.preimage (fun x_1 => HMul.hMul x x_1) ↑(v.ltAddSubgroup  …
        -/
        change v (x * y) < _
        /-
          case h
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ : Units Γ₀
          Hx : Eq (v x) 0
          y : R
          a✝ : Membership.mem (↑(v.ltAddSubgroup 1)) y
          ⊢ LT.lt (v (HMul.hMul x y)) ↑γ
        -/
        rw [Valuation.map_mul, Hx, zero_mul]
        /-
          case h
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ : Units Γ₀
          Hx : Eq (v x) 0
          y : R
          a✝ : Membership.mem (↑(v.ltAddSubgroup 1)) y
          ⊢ LT.lt 0 ↑γ
        -/
        exact Units.zero_lt γ
        /-
          🎉 no goals
        -/
        /-
          case inr.intro
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ γx : Units Γ₀
          Hx : Eq (v x) ↑γx
          ⊢ Exists fun j => HasSubset.Subset (↑(v.ltAddSubgroup j)) (Set.preimage (fun x …
        -/
      · use γx⁻¹ * γ
        /-
          case h
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ γx : Units Γ₀
          Hx : Eq (v x) ↑γx
          ⊢ HasSubset.Subset (↑(v.ltAddSubgroup (HMul.hMul (Inv.inv γx) γ))) (Set.preima …
        -/
        rintro y (vy_lt : v y < ↑(γx⁻¹ * γ))
        /-
          case h
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ γx : Units Γ₀
          Hx : Eq (v x) ↑γx
          y : R
          vy_lt : LT.lt (v y) ↑(HMul.hMul (Inv.inv γx) γ)
          ⊢ Membership.mem (Set.preimage (fun x_1 => HMul.hMul x x_1) ↑(v.ltAddSubgroup  …
        -/
        change (v (x * y) : Γ₀) < γ
        /-
          case h
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ γx : Units Γ₀
          Hx : Eq (v x) ↑γx
          y : R
          vy_lt : LT.lt (v y) ↑(HMul.hMul (Inv.inv γx) γ)
          ⊢ LT.lt (v (HMul.hMul x y)) ↑γ
        -/
        rw [Valuation.map_mul, Hx, mul_comm]
        /-
          case h
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ γx : Units Γ₀
          Hx : Eq (v x) ↑γx
          y : R
          vy_lt : LT.lt (v y) ↑(HMul.hMul (Inv.inv γx) γ)
          ⊢ LT.lt (HMul.hMul (v y) ↑γx) ↑γ
        -/
        rw [Units.val_mul, mul_comm] at vy_lt
        /-
          case h
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ γx : Units Γ₀
          Hx : Eq (v x) ↑γx
          y : R
          vy_lt : LT.lt (v y) (HMul.hMul ↑γ ↑(Inv.inv γx))
          ⊢ LT.lt (HMul.hMul (v y) ↑γx) ↑γ
        -/
        simpa using mul_inv_lt_of_lt_mul₀ vy_lt
        /-
          🎉 no goals
        -/
    rightMul := by
      /-
        R : Type u
        inst✝¹ : Ring R
        Γ₀ : Type v
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation R Γ₀
        ⊢ ∀ (x : R) (i : Units Γ₀), Exists fun j => HasSubset.Subset (↑(v.ltAddSubgrou …
      -/
      rintro x γ
      /-
        R : Type u
        inst✝¹ : Ring R
        Γ₀ : Type v
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation R Γ₀
        x : R
        γ : Units Γ₀
        ⊢ Exists fun j => HasSubset.Subset (↑(v.ltAddSubgroup j)) (Set.preimage (fun x …
      -/
      rcases GroupWithZero.eq_zero_or_unit (v x) with (Hx | ⟨γx, Hx⟩)
        /-
          case inl
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ : Units Γ₀
          Hx : Eq (v x) 0
          ⊢ Exists fun j => HasSubset.Subset (↑(v.ltAddSubgroup j)) (Set.preimage (fun x …
        -/
      · use 1
        /-
          case h
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ : Units Γ₀
          Hx : Eq (v x) 0
          ⊢ HasSubset.Subset (↑(v.ltAddSubgroup 1)) (Set.preimage (fun x_1 => HMul.hMul  …
        -/
        rintro y _
        /-
          case h
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ : Units Γ₀
          Hx : Eq (v x) 0
          y : R
          a✝ : Membership.mem (↑(v.ltAddSubgroup 1)) y
          ⊢ Membership.mem (Set.preimage (fun x_1 => HMul.hMul x_1 x) ↑(v.ltAddSubgroup  …
        -/
        change v (y * x) < _
        /-
          case h
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ : Units Γ₀
          Hx : Eq (v x) 0
          y : R
          a✝ : Membership.mem (↑(v.ltAddSubgroup 1)) y
          ⊢ LT.lt (v (HMul.hMul y x)) ↑γ
        -/
        rw [Valuation.map_mul, Hx, mul_zero]
        /-
          case h
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ : Units Γ₀
          Hx : Eq (v x) 0
          y : R
          a✝ : Membership.mem (↑(v.ltAddSubgroup 1)) y
          ⊢ LT.lt 0 ↑γ
        -/
        exact Units.zero_lt γ
        /-
          🎉 no goals
        -/
        /-
          case inr.intro
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ γx : Units Γ₀
          Hx : Eq (v x) ↑γx
          ⊢ Exists fun j => HasSubset.Subset (↑(v.ltAddSubgroup j)) (Set.preimage (fun x …
        -/
      · use γx⁻¹ * γ
        /-
          case h
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ γx : Units Γ₀
          Hx : Eq (v x) ↑γx
          ⊢ HasSubset.Subset (↑(v.ltAddSubgroup (HMul.hMul (Inv.inv γx) γ))) (Set.preima …
        -/
        rintro y (vy_lt : v y < ↑(γx⁻¹ * γ))
        /-
          case h
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ γx : Units Γ₀
          Hx : Eq (v x) ↑γx
          y : R
          vy_lt : LT.lt (v y) ↑(HMul.hMul (Inv.inv γx) γ)
          ⊢ Membership.mem (Set.preimage (fun x_1 => HMul.hMul x_1 x) ↑(v.ltAddSubgroup  …
        -/
        change (v (y * x) : Γ₀) < γ
        /-
          case h
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ γx : Units Γ₀
          Hx : Eq (v x) ↑γx
          y : R
          vy_lt : LT.lt (v y) ↑(HMul.hMul (Inv.inv γx) γ)
          ⊢ LT.lt (v (HMul.hMul y x)) ↑γ
        -/
        rw [Valuation.map_mul, Hx]
        /-
          case h
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ γx : Units Γ₀
          Hx : Eq (v x) ↑γx
          y : R
          vy_lt : LT.lt (v y) ↑(HMul.hMul (Inv.inv γx) γ)
          ⊢ LT.lt (HMul.hMul (v y) ↑γx) ↑γ
        -/
        rw [Units.val_mul, mul_comm] at vy_lt
        /-
          case h
          R : Type u
          inst✝¹ : Ring R
          Γ₀ : Type v
          inst✝ : LinearOrderedCommGroupWithZero Γ₀
          v : Valuation R Γ₀
          x : R
          γ γx : Units Γ₀
          Hx : Eq (v x) ↑γx
          y : R
          vy_lt : LT.lt (v y) (HMul.hMul ↑γ ↑(Inv.inv γx))
          ⊢ LT.lt (HMul.hMul (v y) ↑γx) ↑γ
        -/
        simpa using mul_inv_lt_of_lt_mul₀ vy_lt }
        /-
          🎉 no goals
        -/


/-- A valued ring is a ring that comes equipped with a distinguished valuation. The class `Valued`
is designed for the situation that there is a canonical valuation on the ring.

TODO: show that there always exists an equivalent valuation taking values in a type belonging to
the same universe as the ring.

See Note [forgetful inheritance] for why we extend `UniformSpace`, `UniformAddGroup`. -/
class Valued (R : Type u) [Ring R] (Γ₀ : outParam (Type v))
  [LinearOrderedCommGroupWithZero Γ₀] extends UniformSpace R, UniformAddGroup R where
  v : Valuation R Γ₀
  is_topological_valuation : ∀ s, s ∈ 𝓝 (0 : R) ↔ ∃ γ : Γ₀ˣ, { x : R | v x < γ } ⊆ s


/-- Alternative `Valued` constructor for use when there is no preferred `UniformSpace` structure. -/
def mk' (v : Valuation R Γ₀) : Valued R Γ₀ :=
  { v
    toUniformSpace := @TopologicalAddGroup.toUniformSpace R _ v.subgroups_basis.topology _
    toUniformAddGroup := @comm_topologicalAddGroup_is_uniform _ _ v.subgroups_basis.topology _
    is_topological_valuation := by
      /-
        R : Type u
        inst✝¹ : Ring R
        Γ₀ : Type v
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation R Γ₀
        ⊢ ∀ (s : Set R), Iff (Membership.mem (nhds 0) s) (Exists fun γ => HasSubset.Su …
      -/
      letI := @TopologicalAddGroup.toUniformSpace R _ v.subgroups_basis.topology _
      /-
        R : Type u
        inst✝¹ : Ring R
        Γ₀ : Type v
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation R Γ₀
        this : UniformSpace R := TopologicalAddGroup.toUniformSpace R
        ⊢ ∀ (s : Set R), Iff (Membership.mem (nhds 0) s) (Exists fun γ => HasSubset.Su …
      -/
      intro s
      /-
        R : Type u
        inst✝¹ : Ring R
        Γ₀ : Type v
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation R Γ₀
        this : UniformSpace R := TopologicalAddGroup.toUniformSpace R
        s : Set R
        ⊢ Iff (Membership.mem (nhds 0) s) (Exists fun γ => HasSubset.Subset (setOf fun …
      -/
      rw [Filter.hasBasis_iff.mp v.subgroups_basis.hasBasis_nhds_zero s]
      /-
        R : Type u
        inst✝¹ : Ring R
        Γ₀ : Type v
        inst✝ : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation R Γ₀
        this : UniformSpace R := TopologicalAddGroup.toUniformSpace R
        s : Set R
        ⊢ Iff (Exists fun i => And True (HasSubset.Subset (↑(v.ltAddSubgroup i)) s)) ( …
      -/
      exact exists_congr fun γ => by rw [true_and]; rfl }
      /-
        🎉 no goals
      -/


theorem hasBasis_nhds_zero :
    (𝓝 (0 : R)).HasBasis (fun _ => True) fun γ : Γ₀ˣ => { x | v x < (γ : Γ₀) } := by
  /-
    R : Type u
    inst✝¹ : Ring R
    Γ₀ : Type v
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    _i : Valued R Γ₀
    ⊢ (nhds 0).HasBasis (fun x => True) fun γ => setOf fun x => LT.lt (Valued.v x) …
  -/
  simp [Filter.hasBasis_iff, is_topological_valuation]
  /-
    🎉 no goals
  -/

-- Porting note: Replaced `𝓤 R` with `uniformity R`

theorem hasBasis_uniformity : (uniformity R).HasBasis (fun _ => True)
    fun γ : Γ₀ˣ => { p : R × R | v (p.2 - p.1) < (γ : Γ₀) } := by
  /-
    R : Type u
    inst✝¹ : Ring R
    Γ₀ : Type v
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    _i : Valued R Γ₀
    ⊢ (uniformity R).HasBasis (fun x => True) fun γ => setOf fun p => LT.lt (Value …
  -/
  rw [uniformity_eq_comap_nhds_zero]
  /-
    R : Type u
    inst✝¹ : Ring R
    Γ₀ : Type v
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    _i : Valued R Γ₀
    ⊢ (Filter.comap (fun x => HSub.hSub x.2 x.1) (nhds 0)).HasBasis (fun x => True …
  -/
  exact (hasBasis_nhds_zero R Γ₀).comap _
  /-
    🎉 no goals
  -/


theorem toUniformSpace_eq :
    toUniformSpace = @TopologicalAddGroup.toUniformSpace R _ v.subgroups_basis.topology _ :=
  UniformSpace.ext
    ((hasBasis_uniformity R Γ₀).eq_of_same_basis <| v.subgroups_basis.hasBasis_nhds_zero.comap _)


theorem mem_nhds {s : Set R} {x : R} : s ∈ 𝓝 x ↔ ∃ γ : Γ₀ˣ, { y | (v (y - x) : Γ₀) < γ } ⊆ s := by
  simp only [← nhds_translation_add_neg x, ← sub_eq_add_neg, preimage_setOf_eq, true_and,
    ((hasBasis_nhds_zero R Γ₀).comap fun y => y - x).mem_iff]


theorem mem_nhds_zero {s : Set R} : s ∈ 𝓝 (0 : R) ↔ ∃ γ : Γ₀ˣ, { x | v x < (γ : Γ₀) } ⊆ s := by
  /-
    R : Type u
    inst✝¹ : Ring R
    Γ₀ : Type v
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    _i : Valued R Γ₀
    s : Set R
    ⊢ Iff (Membership.mem (nhds 0) s) (Exists fun γ => HasSubset.Subset (setOf fun …
  -/
  simp only [mem_nhds, sub_zero]
  /-
    🎉 no goals
  -/


theorem loc_const {x : R} (h : (v x : Γ₀) ≠ 0) : { y : R | v y = v x } ∈ 𝓝 x := by
  /-
    R : Type u
    inst✝¹ : Ring R
    Γ₀ : Type v
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    _i : Valued R Γ₀
    x : R
    h : Ne (Valued.v x) 0
    ⊢ Membership.mem (nhds x) (setOf fun y => Eq (Valued.v y) (Valued.v x))
  -/
  rw [mem_nhds]
  /-
    R : Type u
    inst✝¹ : Ring R
    Γ₀ : Type v
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    _i : Valued R Γ₀
    x : R
    h : Ne (Valued.v x) 0
    ⊢ Exists fun γ => HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub  …
  -/
  use Units.mk0 _ h
  /-
    case h
    R : Type u
    inst✝¹ : Ring R
    Γ₀ : Type v
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    _i : Valued R Γ₀
    x : R
    h : Ne (Valued.v x) 0
    ⊢ HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y x)) ↑(Units.mk …
  -/
  rw [Units.val_mk0]
  /-
    case h
    R : Type u
    inst✝¹ : Ring R
    Γ₀ : Type v
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    _i : Valued R Γ₀
    x : R
    h : Ne (Valued.v x) 0
    ⊢ HasSubset.Subset (setOf fun y => LT.lt (Valued.v (HSub.hSub y x)) (Valued.v  …
  -/
  intro y y_in
  /-
    case h
    R : Type u
    inst✝¹ : Ring R
    Γ₀ : Type v
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    _i : Valued R Γ₀
    x : R
    h : Ne (Valued.v x) 0
    y : R
    y_in : Membership.mem (setOf fun y => LT.lt (Valued.v (HSub.hSub y x)) (Valued …
    ⊢ Membership.mem (setOf fun y => Eq (Valued.v y) (Valued.v x)) y
  -/
  exact Valuation.map_eq_of_sub_lt _ y_in
  /-
    🎉 no goals
  -/


instance (priority := 100) : TopologicalRing R :=
  (toUniformSpace_eq R Γ₀).symm ▸ v.subgroups_basis.toRingFilterBasis.isTopologicalRing


theorem cauchy_iff {F : Filter R} : Cauchy F ↔
    F.NeBot ∧ ∀ γ : Γ₀ˣ, ∃ M ∈ F, ∀ᵉ (x ∈ M) (y ∈ M), (v (y - x) : Γ₀) < γ := by
  /-
    R : Type u
    inst✝¹ : Ring R
    Γ₀ : Type v
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    _i : Valued R Γ₀
    F : Filter R
    ⊢ Iff (Cauchy F) (And F.NeBot (∀ (γ : Units Γ₀), Exists fun M => And (Membersh …
  -/
  rw [toUniformSpace_eq, AddGroupFilterBasis.cauchy_iff]
  /-
    R : Type u
    inst✝¹ : Ring R
    Γ₀ : Type v
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    _i : Valued R Γ₀
    F : Filter R
    ⊢ Iff (And F.NeBot (∀ (U : Set R), Membership.mem RingFilterBasis.toAddGroupFi …
  -/
  apply and_congr Iff.rfl
  /-
    R : Type u
    inst✝¹ : Ring R
    Γ₀ : Type v
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    _i : Valued R Γ₀
    F : Filter R
    ⊢ Iff (∀ (U : Set R), Membership.mem RingFilterBasis.toAddGroupFilterBasis U → …
  -/
  simp_rw [Valued.v.subgroups_basis.mem_addGroupFilterBasis_iff]
  /-
    R : Type u
    inst✝¹ : Ring R
    Γ₀ : Type v
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    _i : Valued R Γ₀
    F : Filter R
    ⊢ Iff (∀ (U : Set R), (Exists fun i => Eq U ↑(Valued.v.ltAddSubgroup i)) → Exi …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝¹ : Ring R
      Γ₀ : Type v
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      _i : Valued R Γ₀
      F : Filter R
      ⊢ (∀ (U : Set R), (Exists fun i => Eq U ↑(Valued.v.ltAddSubgroup i)) → Exists  …
    -/
  · intro h γ
    /-
      case mp
      R : Type u
      inst✝¹ : Ring R
      Γ₀ : Type v
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      _i : Valued R Γ₀
      F : Filter R
      h : ∀ (U : Set R), (Exists fun i => Eq U ↑(Valued.v.ltAddSubgroup i)) → Exists …
      γ : Units Γ₀
      ⊢ Exists fun M => And (Membership.mem F M) (∀ (x : R), Membership.mem M x → ∀  …
    -/
    exact h _ (Valued.v.subgroups_basis.mem_addGroupFilterBasis _)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝¹ : Ring R
      Γ₀ : Type v
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      _i : Valued R Γ₀
      F : Filter R
      ⊢ (∀ (γ : Units Γ₀), Exists fun M => And (Membership.mem F M) (∀ (x : R), Memb …
    -/
  · rintro h - ⟨γ, rfl⟩
    /-
      case mpr.intro
      R : Type u
      inst✝¹ : Ring R
      Γ₀ : Type v
      inst✝ : LinearOrderedCommGroupWithZero Γ₀
      _i : Valued R Γ₀
      F : Filter R
      h : ∀ (γ : Units Γ₀), Exists fun M => And (Membership.mem F M) (∀ (x : R), Mem …
      γ : Units Γ₀
      ⊢ Exists fun M => And (Membership.mem F M) (∀ (x : R), Membership.mem M x → ∀  …
    -/
    exact h γ
    /-
      🎉 no goals
    -/


/-- The unit ball of a valued ring is open. -/
theorem integer_isOpen : IsOpen (_i.v.integer : Set R) := by
  /-
    R : Type u
    inst✝¹ : Ring R
    Γ₀ : Type v
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    _i : Valued R Γ₀
    ⊢ IsOpen ↑Valued.v.integer
  -/
  rw [isOpen_iff_mem_nhds]
  /-
    R : Type u
    inst✝¹ : Ring R
    Γ₀ : Type v
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    _i : Valued R Γ₀
    ⊢ ∀ (x : R), Membership.mem (↑Valued.v.integer) x → Membership.mem (nhds x) ↑V …
  -/
  intro x hx
  /-
    R : Type u
    inst✝¹ : Ring R
    Γ₀ : Type v
    inst✝ : LinearOrderedCommGroupWithZero Γ₀
    _i : Valued R Γ₀
    x : R
    hx : Membership.mem (↑Valued.v.integer) x
    ⊢ Membership.mem (nhds x) ↑Valued.v.integer
  -/
  rw [mem_nhds]
  exact ⟨1,
    fun y hy => (sub_add_cancel y x).symm ▸ le_trans (map_add _ _ _) (max_le (le_of_lt hy) hx)⟩


/-- The valuation subring of a valued field is open. -/
theorem valuationSubring_isOpen (K : Type u) [Field K] [hv : Valued K Γ₀] :
    IsOpen (hv.v.valuationSubring : Set K) :=
  integer_isOpen K



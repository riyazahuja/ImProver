/-- This function sends finite cardinals to the corresponding natural, and infinite cardinals
  to `⊤`. -/
noncomputable def toPartENat : Cardinal →+o PartENat :=
  .comp
    { (PartENat.withTopAddEquiv.symm : ℕ∞ →+ PartENat),
      (PartENat.withTopOrderIso.symm : ℕ∞ →o PartENat) with }
    toENat


@[simp]
theorem partENatOfENat_toENat (c : Cardinal) : (toENat c : PartENat) = toPartENat c := rfl


@[simp]
theorem toPartENat_natCast (n : ℕ) : toPartENat n = n := by
  /-
    n : Nat
    ⊢ Eq (Cardinal.toPartENat ↑n) ↑n
  -/
  simp only [← partENatOfENat_toENat, toENat_nat, PartENat.ofENat_coe]
  /-
    🎉 no goals
  -/


theorem toPartENat_apply_of_lt_aleph0 {c : Cardinal} (h : c < ℵ₀) : toPartENat c = toNat c := by
  /-
    c : Cardinal.{u_1}
    h : LT.lt c Cardinal.aleph0
    ⊢ Eq (Cardinal.toPartENat c) ↑(Cardinal.toNat c)
  -/
  lift c to ℕ using h; simp
                       /-
                         🎉 no goals
                       -/


theorem toPartENat_eq_top {c : Cardinal} :
    toPartENat c = ⊤ ↔ ℵ₀ ≤ c := by
  rw [← partENatOfENat_toENat, ← PartENat.withTopEquiv_symm_top, ← toENat_eq_top,
    ← PartENat.withTopEquiv.symm.injective.eq_iff]
  /-
    c : Cardinal.{u_1}
    ⊢ Iff (Eq (↑(Cardinal.toENat c)) (PartENat.withTopEquiv.symm Top.top)) (Eq (Pa …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem toPartENat_apply_of_aleph0_le {c : Cardinal} (h : ℵ₀ ≤ c) : toPartENat c = ⊤ :=
  congr_arg PartENat.ofENat (toENat_eq_top.2 h)


@[deprecated (since := "2024-02-15")]
alias toPartENat_cast := toPartENat_natCast


@[simp]
theorem mk_toPartENat_of_infinite [h : Infinite α] : toPartENat #α = ⊤ :=
  toPartENat_apply_of_aleph0_le (infinite_iff.1 h)


@[simp]
theorem aleph0_toPartENat : toPartENat ℵ₀ = ⊤ :=
  toPartENat_apply_of_aleph0_le le_rfl


theorem toPartENat_surjective : Surjective toPartENat := fun x =>
  PartENat.casesOn x ⟨ℵ₀, toPartENat_apply_of_aleph0_le le_rfl⟩ fun n => ⟨n, toPartENat_natCast n⟩


@[deprecated (since := "2024-02-15")] alias toPartENat_eq_top_iff_le_aleph0 := toPartENat_eq_top


theorem toPartENat_strictMonoOn : StrictMonoOn toPartENat (Set.Iic ℵ₀) :=
  PartENat.withTopOrderIso.symm.strictMono.comp_strictMonoOn toENat_strictMonoOn


lemma toPartENat_le_iff_of_le_aleph0 {c c' : Cardinal} (h : c ≤ ℵ₀) :
    toPartENat c ≤ toPartENat c' ↔ c ≤ c' := by
  /-
    c c' : Cardinal.{u_1}
    h : LE.le c Cardinal.aleph0
    ⊢ Iff (LE.le (Cardinal.toPartENat c) (Cardinal.toPartENat c')) (LE.le c c')
  -/
  lift c to ℕ∞ using h
  simp_rw [← partENatOfENat_toENat, toENat_ofENat, enat_gc _,
   ← PartENat.withTopOrderIso.symm.le_iff_le, PartENat.ofENat_le, map_le_map_iff]


lemma toPartENat_le_iff_of_lt_aleph0 {c c' : Cardinal} (hc' : c' < ℵ₀) :
    toPartENat c ≤ toPartENat c' ↔ c ≤ c' := by
  /-
    c c' : Cardinal.{u_1}
    hc' : LT.lt c' Cardinal.aleph0
    ⊢ Iff (LE.le (Cardinal.toPartENat c) (Cardinal.toPartENat c')) (LE.le c c')
  -/
  lift c' to ℕ using hc'
  simp_rw [← partENatOfENat_toENat, toENat_nat, ← toENat_le_nat,
   ← PartENat.withTopOrderIso.symm.le_iff_le, PartENat.ofENat_le, map_le_map_iff]


lemma toPartENat_inj_of_le_aleph0 {c c' : Cardinal} (hc : c ≤ ℵ₀) (hc' : c' ≤ ℵ₀) :
    toPartENat c = toPartENat c' ↔ c = c' :=
  toPartENat_strictMonoOn.injOn.eq_iff hc hc'


@[deprecated (since := "2024-12-29")] alias toPartENat_eq_iff_of_le_aleph0 :=
  toPartENat_inj_of_le_aleph0


theorem toPartENat_mono {c c' : Cardinal} (h : c ≤ c') :
    toPartENat c ≤ toPartENat c' :=
  OrderHomClass.mono _ h


theorem toPartENat_lift (c : Cardinal.{v}) : toPartENat (lift.{u, v} c) = toPartENat c := by
  /-
    c : Cardinal.{v}
    ⊢ Eq (Cardinal.toPartENat (Cardinal.lift.{u, v} c)) (Cardinal.toPartENat c)
  -/
  simp only [← partENatOfENat_toENat, toENat_lift]
  /-
    🎉 no goals
  -/


theorem toPartENat_congr {β : Type v} (e : α ≃ β) : toPartENat #α = toPartENat #β := by
  /-
    α : Type u
    β : Type v
    e : Equiv α β
    ⊢ Eq (Cardinal.toPartENat (Cardinal.mk α)) (Cardinal.toPartENat (Cardinal.mk β))
  -/
  rw [← toPartENat_lift, lift_mk_eq.{_, _,v}.mpr ⟨e⟩, toPartENat_lift]
  /-
    🎉 no goals
  -/


theorem mk_toPartENat_eq_coe_card [Fintype α] : toPartENat #α = Fintype.card α := by
  /-
    α : Type u
    inst✝ : Fintype α
    ⊢ Eq (Cardinal.toPartENat (Cardinal.mk α)) ↑(Fintype.card α)
  -/
  simp
  /-
    🎉 no goals
  -/



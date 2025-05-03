/-- The `R`-subalgebra of `S`-integers of `K`. -/
@[simps!]
def integer : Subalgebra R K :=
  {
    (⨅ (v) (_ : v ∉ S), (v : HeightOneSpectrum R).valuation.valuationSubring.toSubring).copy
        {x : K | ∀ (v) (_ : v ∉ S), (v : HeightOneSpectrum R).valuation x ≤ 1} <|
                          /-
                            R : Type u
                            inst✝⁴ : CommRing R
                            inst✝³ : IsDedekindDomain R
                            S : Set (IsDedekindDomain.HeightOneSpectrum R)
                            K : Type v
                            inst✝² : Field K
                            inst✝¹ : Algebra R K
                            inst✝ : IsFractionRing R K
                            x✝ : K
                            ⊢ Iff (Membership.mem (setOf fun x => ∀ (v : IsDedekindDomain.HeightOneSpectru …
                          -/
      Set.ext fun _ => by simp [SetLike.mem_coe, Subring.mem_iInf] with
                          /-
                            🎉 no goals
                          -/
    algebraMap_mem' := fun x v _ => v.valuation_le_one x }


theorem integer_eq :
    (S.integer K).toSubring =
      ⨅ (v) (_ : v ∉ S), (v : HeightOneSpectrum R).valuation.valuationSubring.toSubring :=
                     /-
                       R : Type u
                       inst✝⁴ : CommRing R
                       inst✝³ : IsDedekindDomain R
                       S : Set (IsDedekindDomain.HeightOneSpectrum R)
                       K : Type v
                       inst✝² : Field K
                       inst✝¹ : Algebra R K
                       inst✝ : IsFractionRing R K
                       ⊢ Eq ↑(S.integer K).toSubring ↑(iInf fun v => iInf fun x => v.valuation.valuat …
                     -/
  SetLike.ext' <| by ext; simp
                          /-
                            🎉 no goals
                          -/


theorem integer_valuation_le_one (x : S.integer K) {v : HeightOneSpectrum R} (hv : v ∉ S) :
    v.valuation (x : K) ≤ 1 :=
  x.property v hv


/-- The subgroup of `S`-units of `Kˣ`. -/
@[simps!]
def unit : Subgroup Kˣ :=
  (⨅ (v) (_ : v ∉ S), (v : HeightOneSpectrum R).valuation.valuationSubring.unitGroup).copy
      {x : Kˣ | ∀ (v) (_ : v ∉ S), (v : HeightOneSpectrum R).valuation (x : K) = 1} <|
    Set.ext fun _ => by
      -- Porting note: was
      -- simpa only [SetLike.mem_coe, Subgroup.mem_iInf, Valuation.mem_unitGroup_iff]
      /-
        R : Type u
        inst✝⁴ : CommRing R
        inst✝³ : IsDedekindDomain R
        S : Set (IsDedekindDomain.HeightOneSpectrum R)
        K : Type v
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsFractionRing R K
        x✝ : Units K
        ⊢ Iff (Membership.mem (setOf fun x => ∀ (v : IsDedekindDomain.HeightOneSpectru …
      -/
      simp only [mem_setOf, SetLike.mem_coe, Subgroup.mem_iInf, Valuation.mem_unitGroup_iff]
      /-
        🎉 no goals
      -/


theorem unit_eq :
    S.unit K = ⨅ (v) (_ : v ∉ S), (v : HeightOneSpectrum R).valuation.valuationSubring.unitGroup :=
  Subgroup.copy_eq _ _ _


theorem unit_valuation_eq_one (x : S.unit K) {v : HeightOneSpectrum R} (hv : v ∉ S) :
    v.valuation ((x : Kˣ) : K) = 1 :=
  x.property v hv


/-- The group of `S`-units is the group of units of the ring of `S`-integers. -/
@[simps apply_val_coe symm_apply_coe]
def unitEquivUnitsInteger : S.unit K ≃* (S.integer K)ˣ where
  toFun x :=
    ⟨⟨((x : Kˣ) : K), fun v hv => (x.property v hv).le⟩,
      ⟨((x⁻¹ : Kˣ) : K), fun v hv => (x⁻¹.property v hv).le⟩,
      Subtype.ext x.val.val_inv, Subtype.ext x.val.inv_val⟩
  invFun x :=
    ⟨Units.mk0 x fun hx => x.ne_zero (ZeroMemClass.coe_eq_zero.mp hx),
    fun v hv =>
      eq_one_of_one_le_mul_left (x.val.property v hv) (x.inv.property v hv) <|
        Eq.ge <| by
          -- Porting note: was
          -- rw [← map_mul]; convert v.valuation.map_one; exact subtype.mk_eq_mk.mp x.val_inv⟩
          /-
            R : Type u
            inst✝⁴ : CommRing R
            inst✝³ : IsDedekindDomain R
            S : Set (IsDedekindDomain.HeightOneSpectrum R)
            K : Type v
            inst✝² : Field K
            inst✝¹ : Algebra R K
            inst✝ : IsFractionRing R K
            x : Units (Subtype fun x => Membership.mem (S.integer K) x)
            v : IsDedekindDomain.HeightOneSpectrum R
            hv : Not (Membership.mem S v)
            ⊢ Eq (HMul.hMul (v.valuation ↑(Units.mk0 ↑↑x ⋯)) (v.valuation ↑x.inv)) 1
          -/
          rw [Units.val_mk0, ← map_mul, Subtype.mk_eq_mk.mp x.val_inv, v.valuation.map_one]⟩
          /-
            🎉 no goals
          -/
                   /-
                     R : Type u
                     inst✝⁴ : CommRing R
                     inst✝³ : IsDedekindDomain R
                     S : Set (IsDedekindDomain.HeightOneSpectrum R)
                     K : Type v
                     inst✝² : Field K
                     inst✝¹ : Algebra R K
                     inst✝ : IsFractionRing R K
                     x✝ : Subtype fun x => Membership.mem (S.unit K) x
                     ⊢ Eq ((fun x => ⟨Units.mk0 ↑↑x ⋯, ⋯⟩) ((fun x => { val := ⟨↑↑x, ⋯⟩, inv := ⟨↑( …
                   -/
  left_inv _ := by ext; rfl
                        /-
                          🎉 no goals
                        -/
                    /-
                      R : Type u
                      inst✝⁴ : CommRing R
                      inst✝³ : IsDedekindDomain R
                      S : Set (IsDedekindDomain.HeightOneSpectrum R)
                      K : Type v
                      inst✝² : Field K
                      inst✝¹ : Algebra R K
                      inst✝ : IsFractionRing R K
                      x✝ : Units (Subtype fun x => Membership.mem (S.integer K) x)
                      ⊢ Eq ((fun x => { val := ⟨↑↑x, ⋯⟩, inv := ⟨↑(Inv.inv ↑x), ⋯⟩, val_inv := ⋯, in …
                    -/
  right_inv _ := by ext; rfl
                         /-
                           🎉 no goals
                         -/
                     /-
                       R : Type u
                       inst✝⁴ : CommRing R
                       inst✝³ : IsDedekindDomain R
                       S : Set (IsDedekindDomain.HeightOneSpectrum R)
                       K : Type v
                       inst✝² : Field K
                       inst✝¹ : Algebra R K
                       inst✝ : IsFractionRing R K
                       x✝¹ x✝ : Subtype fun x => Membership.mem (S.unit K) x
                       ⊢ Eq ({ toFun := fun x => { val := ⟨↑↑x, ⋯⟩, inv := ⟨↑(Inv.inv ↑x), ⋯⟩, val_in …
                     -/
  map_mul' _ _ := by ext; rfl
                          /-
                            🎉 no goals
                          -/



/-- We can extend a valuation `v` on a ring to a localization at a submonoid of
the complement of `v.supp`. -/
noncomputable def Valuation.extendToLocalization : Valuation B Γ :=
  let f := IsLocalization.toLocalizationMap S B
  let h : ∀ s : S, IsUnit (v.1.toMonoidHom s) := fun s => isUnit_iff_ne_zero.2 (hS s.2)
  { f.lift h with
                    /-
                      A : Type u_1
                      inst✝⁴ : CommRing A
                      Γ : Type u_2
                      inst✝³ : LinearOrderedCommGroupWithZero Γ
                      v : Valuation A Γ
                      S : Submonoid A
                      hS : LE.le S v.supp.primeCompl
                      B : Type u_3
                      inst✝² : CommRing B
                      inst✝¹ : Algebra A B
                      inst✝ : IsLocalization S B
                      f : S.LocalizationMap B := IsLocalization.toLocalizationMap S B
                      h : ∀ (s : Subtype fun x => Membership.mem S x), IsUnit (↑v.toMonoidWithZeroHo …
                      ⊢ Eq ((↑__src✝).toFun 0) 0
                    -/
                                                       /-
                                                         🎉 no goals
                                                       -/
    map_zero' := by convert f.lift_eq (P := Γ) _ 0 <;> simp [f]
                                                       /-
                                                         🎉 no goals
                                                       -/
    map_add_le_max' := fun x y => by
      obtain ⟨a, b, s, rfl, rfl⟩ : ∃ (a b : A) (s : S), f.mk' a s = x ∧ f.mk' b s = y := by
        obtain ⟨a, s, rfl⟩ := f.mk'_surjective x
        obtain ⟨b, t, rfl⟩ := f.mk'_surjective y
        use a * t, b * s, s * t
        constructor <;>
          · rw [f.mk'_eq_iff_eq, Submonoid.coe_mul]
            ring_nf
      /-
        case intro.intro.intro.intro
        A : Type u_1
        inst✝⁴ : CommRing A
        Γ : Type u_2
        inst✝³ : LinearOrderedCommGroupWithZero Γ
        v : Valuation A Γ
        S : Submonoid A
        hS : LE.le S v.supp.primeCompl
        B : Type u_3
        inst✝² : CommRing B
        inst✝¹ : Algebra A B
        inst✝ : IsLocalization S B
        f : S.LocalizationMap B := IsLocalization.toLocalizationMap S B
        h : ∀ (s : Subtype fun x => Membership.mem S x), IsUnit (↑v.toMonoidWithZeroHo …
        a b : A
        s : Subtype fun x => Membership.mem S x
        ⊢ LE.le ((↑{ toFun := (↑__src✝).toFun, map_zero' := ⋯, map_one' := ⋯, map_mul' …
      -/
      convert_to f.lift h (f.mk' (a + b) s) ≤ max (f.lift h _) (f.lift h _)
        /-
          case h.e'_3
          A : Type u_1
          inst✝⁴ : CommRing A
          Γ : Type u_2
          inst✝³ : LinearOrderedCommGroupWithZero Γ
          v : Valuation A Γ
          S : Submonoid A
          hS : LE.le S v.supp.primeCompl
          B : Type u_3
          inst✝² : CommRing B
          inst✝¹ : Algebra A B
          inst✝ : IsLocalization S B
          f : S.LocalizationMap B := IsLocalization.toLocalizationMap S B
          h : ∀ (s : Subtype fun x => Membership.mem S x), IsUnit (↑v.toMonoidWithZeroHo …
          a b : A
          s : Subtype fun x => Membership.mem S x
          ⊢ Eq ((↑{ toFun := (↑__src✝).toFun, map_zero' := ⋯, map_one' := ⋯, map_mul' := …
        -/
      · refine congr_arg (f.lift h) (IsLocalization.eq_mk'_iff_mul_eq.2 ?_)
        /-
          case h.e'_3
          A : Type u_1
          inst✝⁴ : CommRing A
          Γ : Type u_2
          inst✝³ : LinearOrderedCommGroupWithZero Γ
          v : Valuation A Γ
          S : Submonoid A
          hS : LE.le S v.supp.primeCompl
          B : Type u_3
          inst✝² : CommRing B
          inst✝¹ : Algebra A B
          inst✝ : IsLocalization S B
          f : S.LocalizationMap B := IsLocalization.toLocalizationMap S B
          h : ∀ (s : Subtype fun x => Membership.mem S x), IsUnit (↑v.toMonoidWithZeroHo …
          a b : A
          s : Subtype fun x => Membership.mem S x
          ⊢ Eq (HMul.hMul (HAdd.hAdd (f.mk' a s) (f.mk' b s)) ((algebraMap A B) ↑s)) ((a …
        -/
        rw [add_mul, _root_.map_add]
        /-
          case h.e'_3
          A : Type u_1
          inst✝⁴ : CommRing A
          Γ : Type u_2
          inst✝³ : LinearOrderedCommGroupWithZero Γ
          v : Valuation A Γ
          S : Submonoid A
          hS : LE.le S v.supp.primeCompl
          B : Type u_3
          inst✝² : CommRing B
          inst✝¹ : Algebra A B
          inst✝ : IsLocalization S B
          f : S.LocalizationMap B := IsLocalization.toLocalizationMap S B
          h : ∀ (s : Subtype fun x => Membership.mem S x), IsUnit (↑v.toMonoidWithZeroHo …
          a b : A
          s : Subtype fun x => Membership.mem S x
          ⊢ Eq (HAdd.hAdd (HMul.hMul (f.mk' a s) ((algebraMap A B) ↑s)) (HMul.hMul (f.mk …
        -/
        iterate 2 erw [IsLocalization.mk'_spec]
        /-
          🎉 no goals
        -/
      /-
        case intro.intro.intro.intro.convert_3
        A : Type u_1
        inst✝⁴ : CommRing A
        Γ : Type u_2
        inst✝³ : LinearOrderedCommGroupWithZero Γ
        v : Valuation A Γ
        S : Submonoid A
        hS : LE.le S v.supp.primeCompl
        B : Type u_3
        inst✝² : CommRing B
        inst✝¹ : Algebra A B
        inst✝ : IsLocalization S B
        f : S.LocalizationMap B := IsLocalization.toLocalizationMap S B
        h : ∀ (s : Subtype fun x => Membership.mem S x), IsUnit (↑v.toMonoidWithZeroHo …
        a b : A
        s : Subtype fun x => Membership.mem S x
        ⊢ LE.le ((f.lift h) (f.mk' (HAdd.hAdd a b) s)) (Max.max ((f.lift h) (f.mk' a s …
      -/
      iterate 3 rw [f.lift_mk']
      /-
        case intro.intro.intro.intro.convert_3
        A : Type u_1
        inst✝⁴ : CommRing A
        Γ : Type u_2
        inst✝³ : LinearOrderedCommGroupWithZero Γ
        v : Valuation A Γ
        S : Submonoid A
        hS : LE.le S v.supp.primeCompl
        B : Type u_3
        inst✝² : CommRing B
        inst✝¹ : Algebra A B
        inst✝ : IsLocalization S B
        f : S.LocalizationMap B := IsLocalization.toLocalizationMap S B
        h : ∀ (s : Subtype fun x => Membership.mem S x), IsUnit (↑v.toMonoidWithZeroHo …
        a b : A
        s : Subtype fun x => Membership.mem S x
        ⊢ LE.le (HMul.hMul (↑v.toMonoidWithZeroHom (HAdd.hAdd a b)) ↑(Inv.inv ((IsUnit …
      -/
      rw [max_mul_mul_right]
      /-
        case intro.intro.intro.intro.convert_3
        A : Type u_1
        inst✝⁴ : CommRing A
        Γ : Type u_2
        inst✝³ : LinearOrderedCommGroupWithZero Γ
        v : Valuation A Γ
        S : Submonoid A
        hS : LE.le S v.supp.primeCompl
        B : Type u_3
        inst✝² : CommRing B
        inst✝¹ : Algebra A B
        inst✝ : IsLocalization S B
        f : S.LocalizationMap B := IsLocalization.toLocalizationMap S B
        h : ∀ (s : Subtype fun x => Membership.mem S x), IsUnit (↑v.toMonoidWithZeroHo …
        a b : A
        s : Subtype fun x => Membership.mem S x
        ⊢ LE.le (HMul.hMul (↑v.toMonoidWithZeroHom (HAdd.hAdd a b)) ↑(Inv.inv ((IsUnit …
      -/
      apply mul_le_mul_right' (v.map_add a b) }
      /-
        🎉 no goals
      -/


@[simp]
theorem Valuation.extendToLocalization_apply_map_apply (a : A) :
    v.extendToLocalization hS B (algebraMap A B a) = v a :=
  Submonoid.LocalizationMap.lift_eq _ _ a


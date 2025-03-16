/--The complement of the integers in `ℂ`. -/
def Complex.integerComplement := (Set.range ((↑) : ℤ → ℂ))ᶜ


local notation "ℂ_ℤ " => integerComplement


lemma integerComplement_eq : ℂ_ℤ = {z : ℂ | ¬ ∃ (n : ℤ), n = z} := rfl


lemma integerComplement.mem_iff {x : ℂ} : x ∈ ℂ_ℤ ↔ ¬ ∃ (n : ℤ), n = x := Iff.rfl


lemma UpperHalfPlane.coe_mem_integerComplement (z : ℍ) : ↑z ∈ ℂ_ℤ :=
  not_exists.mpr fun x hx ↦ ne_int z x hx.symm


lemma integerComplement.add_coe_int_mem {x : ℂ} (a : ℤ) : x + (a : ℂ) ∈ ℂ_ℤ ↔ x ∈ ℂ_ℤ := by
  /-
    x : Complex
    a : Int
    ⊢ Iff (Membership.mem Complex.integerComplement (HAdd.hAdd x ↑a)) (Membership. …
  -/
  simp only [mem_iff, not_iff_not]
  exact ⟨(Exists.elim · fun n hn ↦ ⟨n - a, by simp [hn]⟩),
    (Exists.elim · fun n hn ↦ ⟨n + a, by simp [hn]⟩)⟩


lemma integerComplement.ne_zero {x : ℂ} (hx : x ∈ ℂ_ℤ) : x ≠ 0 :=
                      /-
                        x : Complex
                        hx : Membership.mem Complex.integerComplement x
                        hx' : Eq x 0
                        ⊢ Eq (↑0) x
                      -/
  fun hx' ↦ hx ⟨0, by exact_mod_cast hx'.symm⟩
                      /-
                        🎉 no goals
                      -/


lemma integerComplement_add_ne_zero {x : ℂ} (hx : x ∈ ℂ_ℤ) (a : ℤ) : x + (a : ℂ)  ≠ 0 :=
  integerComplement.ne_zero ((integerComplement.add_coe_int_mem a).mpr hx)


lemma integerComplement.ne_one {x : ℂ} (hx : x ∈ ℂ_ℤ): x ≠ 1 :=
                      /-
                        x : Complex
                        hx : Membership.mem Complex.integerComplement x
                        hx' : Eq x 1
                        ⊢ Eq (↑1) x
                      -/
  fun hx' ↦ hx ⟨1, by exact_mod_cast hx'.symm⟩
                      /-
                        🎉 no goals
                      -/



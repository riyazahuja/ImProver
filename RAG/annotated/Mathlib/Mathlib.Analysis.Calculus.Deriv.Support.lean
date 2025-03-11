theorem support_deriv_subset : support (deriv f) ⊆ tsupport f := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    E : Type v
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    ⊢ HasSubset.Subset (Function.support (deriv f)) (tsupport f)
  -/
  intro x
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    E : Type v
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    x : 𝕜
    ⊢ Membership.mem (Function.support (deriv f)) x → Membership.mem (tsupport f) x
  -/
  rw [← not_imp_not]
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    E : Type v
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    x : 𝕜
    ⊢ Not (Membership.mem (tsupport f) x) → Not (Membership.mem (Function.support  …
  -/
  intro h2x
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    E : Type v
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    x : 𝕜
    h2x : Not (Membership.mem (tsupport f) x)
    ⊢ Not (Membership.mem (Function.support (deriv f)) x)
  -/
  rw [not_mem_tsupport_iff_eventuallyEq] at h2x
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    E : Type v
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    x : 𝕜
    h2x : (nhds x).EventuallyEq f 0
    ⊢ Not (Membership.mem (Function.support (deriv f)) x)
  -/
  exact nmem_support.mpr (h2x.deriv_eq.trans (deriv_const x 0))
  /-
    🎉 no goals
  -/


protected theorem HasCompactSupport.deriv (hf : HasCompactSupport f) :
    HasCompactSupport (deriv f) :=
  hf.mono' support_deriv_subset



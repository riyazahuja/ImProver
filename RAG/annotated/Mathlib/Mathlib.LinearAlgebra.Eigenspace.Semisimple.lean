lemma apply_eq_of_mem_of_comm_of_isFinitelySemisimple_of_isNil
    {μ : R} {k : ℕ∞} {m : M} (hm : m ∈ f.genEigenspace μ k)
    (hfg : Commute f g) (hss : g.IsFinitelySemisimple) (hnil : IsNilpotent (f - g)) :
    g m = μ • m := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f g : Module.End R M
    μ : R
    k : ENat
    m : M
    hm : Membership.mem ((f.genEigenspace μ) k) m
    hfg : Commute f g
    hss : g.IsFinitelySemisimple
    hnil : IsNilpotent (HSub.hSub f g)
    ⊢ Eq (g m) (HSMul.hSMul μ m)
  -/
  rw [f.mem_genEigenspace] at hm
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f g : Module.End R M
    μ : R
    k : ENat
    m : M
    hm : Exists fun l => And (LE.le (↑l) k) (Membership.mem (LinearMap.ker (HPow.h …
    hfg : Commute f g
    hss : g.IsFinitelySemisimple
    hnil : IsNilpotent (HSub.hSub f g)
    ⊢ Eq (g m) (HSMul.hSMul μ m)
  -/
  obtain ⟨l, -, hm⟩ := hm
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f g : Module.End R M
    μ : R
    k : ENat
    m : M
    hfg : Commute f g
    hss : g.IsFinitelySemisimple
    hnil : IsNilpotent (HSub.hSub f g)
    l : Nat
    hm : Membership.mem (LinearMap.ker (HPow.hPow (HSub.hSub f (HSMul.hSMul μ 1))  …
    ⊢ Eq (g m) (HSMul.hSMul μ m)
  -/
  rw [← f.mem_genEigenspace_nat] at hm
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f g : Module.End R M
    μ : R
    k : ENat
    m : M
    hfg : Commute f g
    hss : g.IsFinitelySemisimple
    hnil : IsNilpotent (HSub.hSub f g)
    l : Nat
    hm : Membership.mem ((f.genEigenspace μ) ↑l) m
    ⊢ Eq (g m) (HSMul.hSMul μ m)
  -/
  set p := f.genEigenspace μ l
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f g : Module.End R M
    μ : R
    k : ENat
    m : M
    hfg : Commute f g
    hss : g.IsFinitelySemisimple
    hnil : IsNilpotent (HSub.hSub f g)
    l : Nat
    p : Submodule R M := (f.genEigenspace μ) ↑l
    hm : Membership.mem p m
    ⊢ Eq (g m) (HSMul.hSMul μ m)
  -/
  have h₁ : MapsTo g p p := mapsTo_genEigenspace_of_comm hfg μ l
  have h₂ : MapsTo (g - algebraMap R (End R M) μ) p p :=
    mapsTo_genEigenspace_of_comm (hfg.sub_right <| Algebra.commute_algebraMap_right μ f) μ l
  have h₃ : MapsTo (f - g) p p :=
    mapsTo_genEigenspace_of_comm (Commute.sub_right rfl hfg) μ l
  have h₄ : MapsTo (f - algebraMap R (End R M) μ) p p :=
    mapsTo_genEigenspace_of_comm (Algebra.mul_sub_algebraMap_commutes f μ) μ l
  replace hfg : Commute (f - algebraMap R (End R M) μ) (f - g) :=
    (Commute.sub_right rfl hfg).sub_left <| Algebra.commute_algebraMap_left μ (f - g)
  suffices IsNilpotent ((g - algebraMap R (End R M) μ).restrict h₂) by
    replace this : g.restrict h₁ - algebraMap R (End R p) μ = 0 :=
      eq_zero_of_isNilpotent_of_isFinitelySemisimple this (by simpa using hss.restrict _)
    simpa [LinearMap.restrict_apply, sub_eq_zero] using LinearMap.congr_fun this ⟨m, hm⟩
  simpa [LinearMap.restrict_sub h₄ h₃] using (LinearMap.restrict_commute hfg h₄ h₃).isNilpotent_sub
    (f.isNilpotent_restrict_sub_algebraMap μ l) (Module.End.isNilpotent.restrict h₃ hnil)


lemma IsFinitelySemisimple.genEigenspace_eq_eigenspace
    (hf : f.IsFinitelySemisimple) (μ : R) {k : ℕ∞} (hk : 0 < k) :
    f.genEigenspace μ k = f.eigenspace μ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : Module.End R M
    hf : f.IsFinitelySemisimple
    μ : R
    k : ENat
    hk : LT.lt 0 k
    ⊢ Eq ((f.genEigenspace μ) k) (f.eigenspace μ)
  -/
  refine le_antisymm (fun m hm ↦ mem_eigenspace_iff.mpr ?_) (f.genEigenspace μ |>.mono ?_)
    /-
      case refine_1
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Module.End R M
      hf : f.IsFinitelySemisimple
      μ : R
      k : ENat
      hk : LT.lt 0 k
      m : M
      hm : Membership.mem ((f.genEigenspace μ) k) m
      ⊢ Eq (f m) (HSMul.hSMul μ m)
    -/
  · apply apply_eq_of_mem_of_comm_of_isFinitelySemisimple_of_isNil hm rfl hf
    /-
      case refine_1
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Module.End R M
      hf : f.IsFinitelySemisimple
      μ : R
      k : ENat
      hk : LT.lt 0 k
      m : M
      hm : Membership.mem ((f.genEigenspace μ) k) m
      ⊢ IsNilpotent (HSub.hSub f f)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : Module.End R M
      hf : f.IsFinitelySemisimple
      μ : R
      k : ENat
      hk : LT.lt 0 k
      ⊢ LE.le 1 k
    -/
  · exact Order.one_le_iff_pos.mpr hk
    /-
      🎉 no goals
    -/


lemma IsFinitelySemisimple.maxGenEigenspace_eq_eigenspace
    (hf : f.IsFinitelySemisimple) (μ : R) :
    f.maxGenEigenspace μ = f.eigenspace μ :=
  hf.genEigenspace_eq_eigenspace μ ENat.top_pos



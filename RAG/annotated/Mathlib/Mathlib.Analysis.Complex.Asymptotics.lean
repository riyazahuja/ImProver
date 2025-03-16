theorem isTheta_ofReal (f : α → ℝ) (l : Filter α) : (f · : α → ℂ) =Θ[l] f :=
                      /-
                        α : Type u_1
                        f : α → Real
                        l : Filter α
                        ⊢ Asymptotics.IsTheta l (fun x => Norm.norm ↑(f x)) f
                      -/
  .of_norm_left <| by simpa using (Asymptotics.isTheta_rfl (f := f)).norm_left
                      /-
                        🎉 no goals
                      -/


@[simp, norm_cast]
theorem isLittleO_ofReal_left {f : α → ℝ} {g : α → E} : (f · : α → ℂ) =o[l] g ↔ f =o[l] g :=
  (isTheta_ofReal f l).isLittleO_congr_left


@[simp, norm_cast]
theorem isLittleO_ofReal_right {f : α → E} {g : α → ℝ} : f =o[l] (g · : α → ℂ) ↔ f =o[l] g :=
  (isTheta_ofReal g l).isLittleO_congr_right


@[simp, norm_cast]
theorem isBigO_ofReal_left {f : α → ℝ} {g : α → E} : (f · : α → ℂ) =O[l] g ↔ f =O[l] g :=
  (isTheta_ofReal f l).isBigO_congr_left


@[simp, norm_cast]
theorem isBigO_ofReal_right {f : α → E} {g : α → ℝ} : f =O[l] (g · : α → ℂ) ↔ f =O[l] g :=
  (isTheta_ofReal g l).isBigO_congr_right


@[simp, norm_cast]
theorem isTheta_ofReal_left {f : α → ℝ} {g : α → E} : (f · : α → ℂ) =Θ[l] g ↔ f =Θ[l] g :=
  (isTheta_ofReal f l).isTheta_congr_left


@[simp, norm_cast]
theorem isTheta_ofReal_right {f : α → E} {g : α → ℝ} : f =Θ[l] (g · : α → ℂ) ↔ f =Θ[l] g :=
  (isTheta_ofReal g l).isTheta_congr_right


lemma isBigO_comp_ofReal_nhds {f g : ℂ → ℂ} {x : ℝ} (h : f =O[𝓝 (x : ℂ)] g) :
    (fun y : ℝ ↦ f y) =O[𝓝 x] (fun y : ℝ ↦ g y) :=
  h.comp_tendsto <| continuous_ofReal.tendsto x


lemma isBigO_comp_ofReal_nhds_ne {f g : ℂ → ℂ} {x : ℝ} (h : f =O[𝓝[≠] (x : ℂ)] g) :
    (fun y : ℝ ↦ f y) =O[𝓝[≠] x] (fun y : ℝ ↦ g y) :=
                                                                                         /-
                                                                                           f g : Complex → Complex
                                                                                           x : Real
                                                                                           h : Asymptotics.IsBigO (nhdsWithin (↑x) (HasCompl.compl (Singleton.singleton ↑ …
                                                                                           x✝¹ : Real
                                                                                           x✝ : Membership.mem (HasCompl.compl (Singleton.singleton x)) x✝¹
                                                                                           ⊢ Membership.mem (HasCompl.compl (Singleton.singleton ↑x)) ↑x✝¹
                                                                                         -/
  h.comp_tendsto <| continuous_ofReal.continuousWithinAt.tendsto_nhdsWithin fun _ _ ↦ by simp_all
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/



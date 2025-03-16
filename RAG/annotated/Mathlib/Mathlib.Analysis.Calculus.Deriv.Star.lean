protected nonrec theorem HasDerivAtFilter.star (h : HasDerivAtFilter f f' x L) :
    HasDerivAtFilter (fun x => star (f x)) (star f') x L := by
  /-
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    f : 𝕜 → F
    inst✝⁴ : StarRing 𝕜
    inst✝³ : TrivialStar 𝕜
    inst✝² : StarAddMonoid F
    inst✝¹ : ContinuousStar F
    inst✝ : StarModule 𝕜 F
    f' : F
    x : 𝕜
    L : Filter 𝕜
    h : HasDerivAtFilter f f' x L
    ⊢ HasDerivAtFilter (fun x => Star.star (f x)) (Star.star f') x L
  -/
  simpa using h.star.hasDerivAtFilter
  /-
    🎉 no goals
  -/


protected nonrec theorem HasDerivWithinAt.star (h : HasDerivWithinAt f f' s x) :
    HasDerivWithinAt (fun x => star (f x)) (star f') s x :=
  h.star


protected nonrec theorem HasDerivAt.star (h : HasDerivAt f f' x) :
    HasDerivAt (fun x => star (f x)) (star f') x :=
  h.star


protected nonrec theorem HasStrictDerivAt.star (h : HasStrictDerivAt f f' x) :
                                                             /-
                                                               𝕜 : Type u
                                                               inst✝⁷ : NontriviallyNormedField 𝕜
                                                               F : Type v
                                                               inst✝⁶ : NormedAddCommGroup F
                                                               inst✝⁵ : NormedSpace 𝕜 F
                                                               f : 𝕜 → F
                                                               inst✝⁴ : StarRing 𝕜
                                                               inst✝³ : TrivialStar 𝕜
                                                               inst✝² : StarAddMonoid F
                                                               inst✝¹ : ContinuousStar F
                                                               inst✝ : StarModule 𝕜 F
                                                               f' : F
                                                               x : 𝕜
                                                               h : HasStrictDerivAt f f' x
                                                               ⊢ HasStrictDerivAt (fun x => Star.star (f x)) (Star.star f') x
                                                             -/
    HasStrictDerivAt (fun x => star (f x)) (star f') x := by simpa using h.star.hasStrictDerivAt
                                                             /-
                                                               🎉 no goals
                                                             -/


protected theorem derivWithin.star (hxs : UniqueDiffWithinAt 𝕜 s x) :
    derivWithin (fun y => star (f y)) s x = star (derivWithin f s x) :=
  DFunLike.congr_fun (fderivWithin_star hxs) _


protected theorem deriv.star : deriv (fun y => star (f y)) x = star (deriv f x) :=
  DFunLike.congr_fun fderiv_star _


@[simp]
protected theorem deriv.star' : (deriv fun y => star (f y)) = fun x => star (deriv f x) :=
  funext fun _ => deriv.star


theorem differentiableOn_id' : DifferentiableOn K (fun x : E => x) s :=
  differentiable_id.differentiableOn


theorem Differentiable.comp' {g : F → G} (hg : Differentiable K g) (hf : Differentiable K f) :
    Differentiable K (fun x => g (f x)) :=
  fun x => DifferentiableAt.comp x (hg (f x)) (hf x)


theorem DifferentiableAt.comp' {f : E → F} {g : F → G} (hg : DifferentiableAt K g (f x))
    (hf : DifferentiableAt K f x) : DifferentiableAt K (fun x => g (f x)) x :=
  (hg.hasFDerivAt.comp x hf.hasFDerivAt).differentiableAt


theorem DifferentiableOn.comp' {g : F → G} {t : Set F} (hg : DifferentiableOn K g t)
    (hf : DifferentiableOn K f s) (st : Set.MapsTo f s t) :
    DifferentiableOn K (fun x => g (f x)) s :=
  fun x hx => DifferentiableWithinAt.comp x (hg (f x) (st hx)) (hf x hx) st



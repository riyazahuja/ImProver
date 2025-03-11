theorem contDiff_id' : ContDiff K n (fun x : E => x) := contDiff_id


theorem contDiffAt_id' : ContDiffAt K n (fun x : E => x) x := contDiffAt_id


theorem contDiffOn_id' : ContDiffOn K n (fun x : E => x) s :=
  contDiff_id.contDiffOn


theorem ContDiff.comp' {g : F → G} (hg : ContDiff K n g) (hf : ContDiff K n f) :
    ContDiff K n (fun x => g (f x)) := ContDiff.comp hg hf


theorem ContDiffAt.comp' {f : E → F} {g : F → G} (hg : ContDiffAt K n g (f x))
    (hf : ContDiffAt K n f x) : ContDiffAt K n (fun x => g (f x)) x := ContDiffAt.comp x hg hf

-- theorem ContDiffOn.comp'' {g : F → G} {t : Set F} (hg : ContDiffOn K n g t)
--     (hf : ContDiffOn K n f s) (st : Set.MapsTo f s t) : ContDiffOn K n (fun x => g (f x)) s :=


theorem contDiff_pi' (hΦ : ∀ i, ContDiff K n fun x => Φ x i) : ContDiff K n Φ :=
  contDiff_pi.2 hΦ


theorem contDiffOn_pi' (hΦ : ∀ i, ContDiffOn K n (fun x => Φ x i) s) : ContDiffOn K n Φ s :=
  contDiffOn_pi.2 hΦ


theorem contDiffAt_pi' (hΦ : ∀ i, ContDiffAt K n (fun x => Φ x i) x) : ContDiffAt K n Φ x :=
  contDiffAt_pi.2 hΦ


theorem ContDiffOn.div' {f g : E → K} {n} (hf : ContDiffOn K n f s)
    (hg : ContDiffOn K n g s) (h₀ : ∀ x ∈ s, g x ≠ 0) : ContDiffOn K n (fun x => f x / g x) s :=
  ContDiffOn.div hf hg h₀



/-- Original version `ContDiff.differentiable_iteratedDeriv` introduces a new variable `(n:ℕ∞)`
and `funProp` can't work with such theorem. The theorem should be state where `n` is explicitly
the smallest possible value i.e. `n=m+1`.

In conjunction with `ContDiff.of_le` we can recover the full power of the original theorem. -/
theorem ContDiff.differentiable_iteratedDeriv' {m : ℕ} {f : K → F}
    (hf : ContDiff K (m+1) f) : Differentiable K (iteratedDeriv m f) :=
  ContDiff.differentiable_iteratedDeriv m hf (Nat.cast_lt.mpr m.lt_succ_self)



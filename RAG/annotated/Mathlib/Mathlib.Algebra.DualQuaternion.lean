/-- The dual quaternions can be equivalently represented as a quaternion with dual coefficients,
or as a dual number with quaternion coefficients.

See also `Matrix.dualNumberEquiv` for a similar result. -/
def dualNumberEquiv : Quaternion (DualNumber R) ≃ₐ[R] DualNumber (Quaternion R) where
  toFun q :=
    (⟨q.re.fst, q.imI.fst, q.imJ.fst, q.imK.fst⟩, ⟨q.re.snd, q.imI.snd, q.imJ.snd, q.imK.snd⟩)
  invFun d :=
    ⟨(d.fst.re, d.snd.re), (d.fst.imI, d.snd.imI), (d.fst.imJ, d.snd.imJ), (d.fst.imK, d.snd.imK)⟩
  left_inv := fun _ => rfl
  right_inv := fun _ => rfl
  map_mul' := by
    /-
      R : Type u_1
      inst✝ : CommRing R
      ⊢ ∀ (x y : Quaternion (DualNumber R)), Eq ({ toFun := fun q => { fst := { re : …
    -/
    intros
    /-
      R : Type u_1
      inst✝ : CommRing R
      x✝ y✝ : Quaternion (DualNumber R)
      ⊢ Eq ({ toFun := fun q => { fst := { re := TrivSqZeroExt.fst q.re, imI := Triv …
    -/
    ext : 1
      /-
        case h1
        R : Type u_1
        inst✝ : CommRing R
        x✝ y✝ : Quaternion (DualNumber R)
        ⊢ Eq (TrivSqZeroExt.fst ({ toFun := fun q => { fst := { re := TrivSqZeroExt.fs …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case h2
        R : Type u_1
        inst✝ : CommRing R
        x✝ y✝ : Quaternion (DualNumber R)
        ⊢ Eq (TrivSqZeroExt.snd ({ toFun := fun q => { fst := { re := TrivSqZeroExt.fs …
      -/
    · dsimp
      /-
        case h2
        R : Type u_1
        inst✝ : CommRing R
        x✝ y✝ : Quaternion (DualNumber R)
        ⊢ Eq { re := TrivSqZeroExt.snd (HMul.hMul x✝ y✝).re, imI := TrivSqZeroExt.snd  …
      -/
                           /-
                             🎉 no goals
                           -/
                           /-
                             🎉 no goals
                           -/
                           /-
                             🎉 no goals
                           -/
      congr 1 <;> simp <;> ring
                           /-
                             🎉 no goals
                           -/
  map_add' := by
    /-
      R : Type u_1
      inst✝ : CommRing R
      ⊢ ∀ (x y : Quaternion (DualNumber R)), Eq ({ toFun := fun q => { fst := { re : …
    -/
    intros
    /-
      R : Type u_1
      inst✝ : CommRing R
      x✝ y✝ : Quaternion (DualNumber R)
      ⊢ Eq ({ toFun := fun q => { fst := { re := TrivSqZeroExt.fst q.re, imI := Triv …
    -/
    rfl
    /-
      🎉 no goals
    -/
  commutes' _ := rfl


@[simp]
theorem re_fst_dualNumberEquiv (q : Quaternion (DualNumber R)) :
    (dualNumberEquiv q).fst.re = q.re.fst :=
  rfl


@[simp]
theorem imI_fst_dualNumberEquiv (q : Quaternion (DualNumber R)) :
    (dualNumberEquiv q).fst.imI = q.imI.fst :=
  rfl


@[simp]
theorem imJ_fst_dualNumberEquiv (q : Quaternion (DualNumber R)) :
    (dualNumberEquiv q).fst.imJ = q.imJ.fst :=
  rfl


@[simp]
theorem imK_fst_dualNumberEquiv (q : Quaternion (DualNumber R)) :
    (dualNumberEquiv q).fst.imK = q.imK.fst :=
  rfl


@[simp]
theorem re_snd_dualNumberEquiv (q : Quaternion (DualNumber R)) :
    (dualNumberEquiv q).snd.re = q.re.snd :=
  rfl


@[simp]
theorem imI_snd_dualNumberEquiv (q : Quaternion (DualNumber R)) :
    (dualNumberEquiv q).snd.imI = q.imI.snd :=
  rfl


@[simp]
theorem imJ_snd_dualNumberEquiv (q : Quaternion (DualNumber R)) :
    (dualNumberEquiv q).snd.imJ = q.imJ.snd :=
  rfl


@[simp]
theorem imK_snd_dualNumberEquiv (q : Quaternion (DualNumber R)) :
    (dualNumberEquiv q).snd.imK = q.imK.snd :=
  rfl


@[simp]
theorem fst_re_dualNumberEquiv_symm (d : DualNumber (Quaternion R)) :
    (dualNumberEquiv.symm d).re.fst = d.fst.re :=
  rfl


@[simp]
theorem fst_imI_dualNumberEquiv_symm (d : DualNumber (Quaternion R)) :
    (dualNumberEquiv.symm d).imI.fst = d.fst.imI :=
  rfl


@[simp]
theorem fst_imJ_dualNumberEquiv_symm (d : DualNumber (Quaternion R)) :
    (dualNumberEquiv.symm d).imJ.fst = d.fst.imJ :=
  rfl


@[simp]
theorem fst_imK_dualNumberEquiv_symm (d : DualNumber (Quaternion R)) :
    (dualNumberEquiv.symm d).imK.fst = d.fst.imK :=
  rfl


@[simp]
theorem snd_re_dualNumberEquiv_symm (d : DualNumber (Quaternion R)) :
    (dualNumberEquiv.symm d).re.snd = d.snd.re :=
  rfl


@[simp]
theorem snd_imI_dualNumberEquiv_symm (d : DualNumber (Quaternion R)) :
    (dualNumberEquiv.symm d).imI.snd = d.snd.imI :=
  rfl


@[simp]
theorem snd_imJ_dualNumberEquiv_symm (d : DualNumber (Quaternion R)) :
    (dualNumberEquiv.symm d).imJ.snd = d.snd.imJ :=
  rfl


@[simp]
theorem snd_imK_dualNumberEquiv_symm (d : DualNumber (Quaternion R)) :
    (dualNumberEquiv.symm d).imK.snd = d.snd.imK :=
  rfl


